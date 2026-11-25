"""Command-line interface: qwen <command>."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import httpx

from qwen_local import QwenError, chat, config, gpu, llama, model, server
from qwen_local.config import Config

GB = 1000**3


def cmd_doctor(cfg: Config) -> int:
    ok = True

    def row(label: str, value: str) -> None:
        print(f"  {label:<12} {value}")

    print("Python")
    row("version", sys.version.split()[0])
    row("interpreter", sys.base_prefix)
    local_python = cfg.root / ".python"
    if not Path(sys.base_prefix).resolve().is_relative_to(local_python.resolve()):
        row("warning", f"not the project-local Python in {local_python}; run .\\setup.ps1")

    print("GPU")
    gpus = gpu.query_gpus()
    if not gpus:
        row("NVIDIA", "not found (llama.cpp will run on the CPU)")
    for g in gpus:
        row(f"[{g.index}]", f"{g.name}, driver {g.driver}")
        row("VRAM", f"{g.total_mib} MiB total, {g.used_mib} MiB used, {g.free_mib} MiB free")
    if cuda := gpu.cuda_driver_version():
        row("CUDA", f"driver supports up to {cuda[0]}.{cuda[1]}")

    print("llama.cpp")
    if cfg.llama.server:
        row("server", f"{cfg.llama.server} (from config)")
    elif installed := llama.read_installed(cfg):
        row("installed", f"{installed.tag} ({installed.backend})")
        row("server", str(installed.server))
    else:
        ok = False
        row("installed", "no; run: qwen install-llama")

    print("Model")
    row("repo", f"{cfg.model.repo}@{cfg.model.revision}")
    for label, path in (("weights", cfg.model.path), ("vision", cfg.model.mmproj_path)):
        if path.is_file():
            row(label, f"{path.name}  {path.stat().st_size / GB:.2f} GB")
        else:
            row(label, f"{path.name}  not downloaded")
            ok = ok and label != "weights"
    cfg.model.dir.mkdir(parents=True, exist_ok=True)
    row("disk free", f"{shutil.disk_usage(cfg.model.dir).free / GB:.0f} GB in {cfg.model.dir}")

    url = cfg.server.base_url
    try:
        response = httpx.get(f"{url}/health", timeout=3)
        state = "ready" if response.status_code == 200 else f"loading ({response.status_code})"
    except httpx.TransportError:
        state = "not running"
    # A running server already holds the VRAM, so a fit estimate would be meaningless.
    if state == "not running" and (report := server.fit_report(cfg, server.ServeOptions())):
        row("fit", report)

    print("Server")
    row("status", f"{state} at {url}")

    print("\nReady to serve." if ok else "\nSetup incomplete; run .\\setup.ps1")
    return 0 if ok else 1


def cmd_setup(cfg: Config, args: argparse.Namespace) -> int:
    print("== llama.cpp")
    llama.install(cfg, upgrade=args.upgrade_llama)
    if not args.skip_model:
        print(f"\n== Model {cfg.model.repo}")
        model.download(cfg, vision=args.vision)
    print()
    return cmd_doctor(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen", description="Run Qwen3.8-27B locally with llama.cpp.")
    parser.add_argument("--config", type=Path, help="config file (default: config.toml in the project)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    sub.add_parser("doctor", help="check the GPU, installed components and the server")

    p = sub.add_parser("setup", help="install llama.cpp and download the model")
    p.add_argument("--vision", action="store_true", help="also download the vision projector")
    p.add_argument("--upgrade-llama", action="store_true", help="replace llama.cpp with the newest build")
    p.add_argument("--skip-model", action="store_true", help="do not download the model")

    p = sub.add_parser("install-llama", help="download a prebuilt llama.cpp")
    p.add_argument("--upgrade", action="store_true", help="replace the installed build with the newest one")
    p.add_argument("--release", help="build tag such as b12728, or 'latest' (default: from config)")
    p.add_argument("--backend", help="auto, cuda, cuda-13.5, cuda-12.4, vulkan or cpu (default: from config)")

    p = sub.add_parser("download", help="download the model from Hugging Face")
    p.add_argument("--vision", action="store_true", help="also download the vision projector")

    p = sub.add_parser(
        "serve",
        help="run llama-server (web UI + OpenAI-compatible API)",
        description="Options not listed here are passed to llama-server unchanged, e.g. --metrics.",
        allow_abbrev=False,
    )
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--ctx", type=int, dest="ctx_size", help="context window in tokens")
    p.add_argument("--gpu-layers", help="layers on the GPU: auto, all, or a number")
    p.add_argument("--vision", action=argparse.BooleanOptionalAction, help="enable image input")
    p.add_argument("--mtp", action=argparse.BooleanOptionalAction, help="speculative decoding with the MTP layer")
    p.add_argument("--effort", choices=config.REASONING_EFFORTS, help="default reasoning effort")
    p.add_argument("--dry-run", action="store_true", help="print the command line and exit")

    p = sub.add_parser("chat", help="chat with the running server in the terminal")
    p.add_argument("-p", "--prompt", help="ask one question and exit ('-' reads stdin)")
    p.add_argument("--no-think", action="store_true", help="answer without thinking")
    p.add_argument("--effort", choices=chat.EFFORTS, help="reasoning effort (default: server setting)")
    p.add_argument("--hide-thinking", action="store_true", help="do not print the reasoning")
    p.add_argument("--system", help="system prompt")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--url", help="server URL (default: from config)")
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for the server to become ready")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # `serve` forwards options it does not know to llama-server. A "--" separator cannot be
    # used for that: PowerShell drops it before the arguments reach this program.
    args, unknown = parser.parse_known_args(argv)
    if unknown and args.command != "serve":
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    try:
        cfg = config.load(args.config)
        match args.command:
            case "doctor":
                return cmd_doctor(cfg)
            case "setup":
                return cmd_setup(cfg, args)
            case "install-llama":
                llama.install(cfg, upgrade=args.upgrade, release=args.release, backend=args.backend)
            case "download":
                model.download(cfg, vision=args.vision)
            case "serve":
                opts = server.ServeOptions(
                    host=args.host,
                    port=args.port,
                    ctx_size=args.ctx_size,
                    gpu_layers=args.gpu_layers,
                    vision=args.vision,
                    mtp=args.mtp,
                    reasoning_effort=args.effort,
                    extra_args=tuple(a for a in unknown if a != "--"),
                )
                return server.serve(cfg, opts, dry_run=args.dry_run)
            case "chat":
                settings = chat.ChatSettings(
                    think=not args.no_think,
                    effort=args.effort,
                    show_thinking=not args.hide_thinking,
                    system=args.system,
                    max_tokens=args.max_tokens,
                )
                return chat.run(cfg, args.url or cfg.server.base_url, settings, prompt=args.prompt, wait=args.wait)
    except QwenError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except httpx.HTTPError as e:
        print(f"error: network: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
