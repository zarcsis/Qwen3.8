"""Build and run the llama-server command line."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from dataclasses import dataclass

from qwen_local import QwenError, gguf, gpu, llama
from qwen_local.config import Config


@dataclass(frozen=True)
class ServeOptions:
    """Command-line overrides; None keeps the value from config.toml."""

    host: str | None = None
    port: int | None = None
    ctx_size: int | None = None
    gpu_layers: str | None = None
    vision: bool | None = None
    mtp: bool | None = None
    reasoning_effort: str | None = None
    extra_args: tuple[str, ...] = ()


def _pick[T](override: T | None, default: T) -> T:
    return default if override is None else override


def build_command(cfg: Config, opts: ServeOptions) -> list[str]:
    s = cfg.server
    m = cfg.model
    if not m.path.is_file():
        raise QwenError(f"Model file missing: {m.path}\nRun: qwen download")

    sampling = cfg.thinking
    cmd = [
        str(llama.server_executable(cfg)),
        "--model", str(m.path),
        "--alias", m.alias,
        "--host", _pick(opts.host, s.host),
        "--port", str(_pick(opts.port, s.port)),
        "--ctx-size", str(_pick(opts.ctx_size, s.ctx_size)),
        "--fit-target", str(s.fit_target_mib),
        "--flash-attn", s.flash_attn,
        "--cache-type-k", s.cache_type_k,
        "--cache-type-v", s.cache_type_v,
        "--parallel", str(s.parallel),
        "--batch-size", str(s.batch_size),
        "--ubatch-size", str(s.ubatch_size),
        "--temp", str(sampling.temperature),
        "--top-p", str(sampling.top_p),
        "--top-k", str(sampling.top_k),
        "--min-p", str(sampling.min_p),
        "--presence-penalty", str(sampling.presence_penalty),
        "--repeat-penalty", str(sampling.repeat_penalty),
    ]  # fmt: skip

    gpu_layers = _pick(opts.gpu_layers, s.gpu_layers)
    if gpu_layers != "auto":
        # An explicit layer count disables automatic fitting of the layers.
        cmd += ["--gpu-layers", gpu_layers]
    if s.threads > 0:
        cmd += ["--threads", str(s.threads)]

    effort = _pick(opts.reasoning_effort, s.reasoning_effort)
    if effort == "off":
        cmd += ["--reasoning", "off"]
    else:
        cmd += ["--reasoning-effort", effort]

    if _pick(opts.vision, s.vision):
        if not m.mmproj_path.is_file():
            raise QwenError(f"Vision projector missing: {m.mmproj_path}\nRun: qwen download --vision")
        cmd += ["--mmproj", str(m.mmproj_path)]
        if not s.mmproj_offload:
            cmd.append("--no-mmproj-offload")
    else:
        cmd.append("--no-mmproj")

    if _pick(opts.mtp, s.mtp):
        # Drafts with the MTP layer stored in the model file itself. A separate MTP GGUF
        # (--model-draft) was no faster in tests and costs another ~1.4 GB of VRAM.
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(s.mtp_draft_max)]

    return cmd + list(s.extra_args) + list(opts.extra_args)


def _port_in_use(host: str, port: int) -> bool:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe, port)) == 0


def fit_report(cfg: Config, opts: ServeOptions) -> str | None:
    """Ask llama.cpp how many layers fit in the free VRAM right now; None if it cannot tell."""
    s, m = cfg.server, cfg.model
    gpus = gpu.query_gpus()
    if not gpus or not m.path.is_file():
        return None
    gpu_layers = _pick(opts.gpu_layers, s.gpu_layers)
    if gpu_layers != "auto":
        return f"GPU layers fixed at {gpu_layers} by configuration."
    try:
        exe = llama.server_executable(cfg).with_name("llama-fit-params.exe")
    except QwenError:
        return None
    if not exe.is_file():
        return None

    # The same memory-relevant settings the server gets. llama-fit-params does not accept the
    # speculative-decoding options, so the MTP draft context is left out of the estimate.
    cmd = [
        str(exe),
        "--model", str(m.path),
        "--ctx-size", str(_pick(opts.ctx_size, s.ctx_size)),
        "--fit-target", str(s.fit_target_mib),
        "--flash-attn", s.flash_attn,
        "--cache-type-k", s.cache_type_k,
        "--cache-type-v", s.cache_type_v,
        "--parallel", str(s.parallel),
        "--batch-size", str(s.batch_size),
        "--ubatch-size", str(s.ubatch_size),
    ]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except OSError, subprocess.TimeoutExpired:
        return None
    match = re.search(r"-ngl (\d+)", result.stdout)
    total = gguf.gpu_layer_count(m.path)
    if result.returncode != 0 or not match or not total:
        return None

    on_gpu = int(match.group(1))
    free = max(g.free_mib for g in gpus)
    if on_gpu >= total:
        line = f"all {total} layers fit on the GPU ({free} MiB VRAM free)."
    else:
        line = (
            f"only {on_gpu} of {total} layers fit on the GPU ({free} MiB VRAM free); the rest run on the "
            "CPU, which is much slower. Close GPU-heavy apps or lower the context size."
        )
    if _pick(opts.mtp, s.mtp):
        line += " MTP needs ~0.6 GB more than this and only speeds things up when that fits too."
    return line


def serve(cfg: Config, opts: ServeOptions, *, dry_run: bool = False) -> int:
    cmd = build_command(cfg, opts)
    print(subprocess.list2cmdline(cmd))
    if dry_run:
        return 0

    host = _pick(opts.host, cfg.server.host)
    port = _pick(opts.port, cfg.server.port)
    if _port_in_use(host, port):
        raise QwenError(f"Port {port} is already in use; is the server already running? Use --port to pick another.")
    if report := fit_report(cfg, opts):
        print(f"Memory check: {report}")
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"\nWeb UI and OpenAI-compatible API: http://{shown_host}:{port}  (API base: /v1, Ctrl+C stops)\n")
    # Print before llama-server's own output, also when stdout is redirected to a file.
    sys.stdout.flush()

    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # The console delivers Ctrl+C to llama-server as well; let it shut down cleanly.
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()
