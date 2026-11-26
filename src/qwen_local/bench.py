"""Benchmark: run llama-server with each profile and measure prompt and generation speed."""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from qwen_local import QwenError, gpu, llama, server
from qwen_local.config import Config

# Server setups to compare; unset fields keep the values from config.toml.
PROFILES: dict[str, server.ServeOptions] = {
    "default": server.ServeOptions(),
    "mtp": server.ServeOptions(mtp=True),
}

GENERATION_PROMPT = "Write a coherent text of about 400 words on how computer memory works."


def _long_prompt(lines: int = 150) -> str:
    """~3,000 tokens of distinct lines (below llama.cpp's 4096-token minimum context)."""
    body = "\n".join(f"Step {i}: register r{i % 32} holds {(i * 7919) % 10007} after the update." for i in range(lines))
    return f"Read the log below, then reply with the single word OK.\n\n{body}"


@dataclass
class ProfileResult:
    profile: str
    ctx_size: int
    mtp: bool
    vram_free_mib: int | None
    prompt_tokens: int
    prompt_tps: float
    gen_tokens: int
    gen_tps: float
    gen_tps_runs: list[float]
    draft_acceptance: float | None


def _chat(client: httpx.Client, base_url: str, cfg: Config, prompt: str, max_tokens: int) -> dict[str, Any]:
    preset = cfg.instruct  # no thinking, so every run generates the same kind of text
    body = {
        "model": cfg.model.alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "presence_penalty": preset.presence_penalty,
        "repeat_penalty": preset.repeat_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
        # Re-process the whole prompt every time instead of reusing the cached prefix.
        "cache_prompt": False,
    }
    response = client.post(f"{base_url}/v1/chat/completions", json=body)
    if response.status_code != 200:
        raise QwenError(f"Server error {response.status_code}: {response.text}")
    return response.json().get("timings", {})


def _wait_ready(client: httpx.Client, base_url: str, proc: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise QwenError(f"llama-server exited with code {proc.returncode}; see {log}")
        try:
            if client.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(1)
    raise QwenError(f"llama-server did not become ready in 10 minutes; see {log}")


def run_profile(
    cfg: Config, name: str, opts: server.ServeOptions, *, runs: int, prompt: str, max_tokens: int, log_dir: Path
) -> ProfileResult:
    s = cfg.server
    host, port = opts.host or s.host, opts.port or s.port
    server.ensure_port_free(host, port)
    base_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    gpus = gpu.query_gpus()
    vram_free = gpus[0].free_mib if gpus else None

    log_path = log_dir / f"server-{name}.log"
    print(f"\n[{name}] starting llama-server (log: {log_path})", flush=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(server.build_command(cfg, opts), stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, read=900.0)) as client:
                _wait_ready(client, base_url, proc, log_path)
                # The context llama.cpp actually picked (it fits it to the VRAM when ctx_size is 0).
                props = client.get(f"{base_url}/props").json()
                ctx_size = int(props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx") or 0)
                print(f"[{name}] context: {ctx_size} tokens")
                _chat(client, base_url, cfg, "Hello", 16)  # warm-up: first requests build the compute graphs
                pp = _chat(client, base_url, cfg, _long_prompt(), 1)
                print(f"[{name}] prompt: {pp.get('prompt_n', 0)} tokens at {pp.get('prompt_per_second', 0):.0f} t/s")
                gens = []
                for i in range(runs):
                    t = _chat(client, base_url, cfg, prompt, max_tokens)
                    gens.append(t)
                    print(f"[{name}] generation {i + 1}/{runs}: {t.get('predicted_per_second', 0):.1f} t/s")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    drafted = sum(t.get("draft_n", 0) for t in gens)
    accepted = sum(t.get("draft_n_accepted", 0) for t in gens)
    speeds = [float(t.get("predicted_per_second", 0)) for t in gens]
    return ProfileResult(
        profile=name,
        ctx_size=ctx_size,
        mtp=s.mtp if opts.mtp is None else opts.mtp,
        vram_free_mib=vram_free,
        prompt_tokens=int(pp.get("prompt_n", 0)),
        prompt_tps=round(float(pp.get("prompt_per_second", 0)), 1),
        gen_tokens=int(statistics.median(t.get("predicted_n", 0) for t in gens)),
        gen_tps=round(statistics.median(speeds), 1),
        gen_tps_runs=[round(v, 1) for v in speeds],
        draft_acceptance=round(accepted / drafted, 3) if drafted else None,
    )


def run(cfg: Config, profiles: list[str], *, runs: int, prompt: str, max_tokens: int, ctx_size: int | None) -> int:
    """Benchmark the profiles; ctx_size fixes the context for all of them (None keeps the config)."""
    if runs < 1:
        raise QwenError("--runs must be at least 1")
    if warning := server.build_warning(cfg):
        print(f"warning: {warning}")

    gpus = gpu.query_gpus()
    installed = llama.read_installed(cfg)
    build = f"{installed.tag} ({installed.backend})" if installed else str(cfg.llama.server)
    hardware = ", ".join(f"{g.name} ({g.total_mib} MiB, compute {g.compute_cap_str})" for g in gpus) or "CPU only"
    print(f"GPU: {hardware}\nDriver: {gpus[0].driver if gpus else '-'}  llama.cpp: {build}  model: {cfg.model.file}")

    out_dir = cfg.root / "runtime" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in profiles:
        opts = PROFILES[name] if ctx_size is None else replace(PROFILES[name], ctx_size=ctx_size)
        results.append(run_profile(cfg, name, opts, runs=runs, prompt=prompt, max_tokens=max_tokens, log_dir=out_dir))

    row = "{:<10} {:>7} {:>4} {:>10} {:>11} {:>8} {:>9}"
    print("\n" + row.format("Profile", "Context", "MTP", "Free VRAM", "Prompt t/s", "Gen t/s", "Accepted"))
    for r in results:
        accepted = f"{r.draft_acceptance:.0%}" if r.draft_acceptance is not None else "-"
        free = f"{r.vram_free_mib} MiB" if r.vram_free_mib is not None else "-"
        mtp = "yes" if r.mtp else "no"
        print(row.format(r.profile, r.ctx_size, mtp, free, f"{r.prompt_tps:.0f}", f"{r.gen_tps:.1f}", accepted))
    print(f"\nGeneration: median of {runs} runs of up to {max_tokens} tokens without thinking.")

    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpus": [{"name": g.name, "vram_mib": g.total_mib, "compute_cap": g.compute_cap_str} for g in gpus],
        "driver": gpus[0].driver if gpus else None,
        "llama_cpp": build,
        "model": f"{cfg.model.repo}/{cfg.model.file}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "results": [asdict(r) for r in results],
    }
    results_file = out_dir / "results.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved to {results_file}")
    return 0
