"""Download the GGUF model files from Hugging Face."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from qwen_local import QwenError
from qwen_local.config import Config

GB = 1000**3


def wanted_files(cfg: Config, *, vision: bool) -> list[str]:
    files = [cfg.model.file]
    if vision:
        files.append(cfg.model.mmproj_file)
    return files


def remote_sizes(cfg: Config, files: list[str]) -> dict[str, int]:
    try:
        infos = HfApi().get_paths_info(cfg.model.repo, files, revision=cfg.model.revision)
    except HfHubHTTPError as e:
        raise QwenError(f"Cannot read {cfg.model.repo} on Hugging Face: {e}") from e
    sizes = {info.path: info.size for info in infos if getattr(info, "size", None) is not None}
    missing = [f for f in files if f not in sizes]
    if missing:
        raise QwenError(f"Not found in {cfg.model.repo}@{cfg.model.revision}: {', '.join(missing)}")
    return sizes


def download(cfg: Config, *, vision: bool = False) -> list[Path]:
    files = wanted_files(cfg, vision=vision)
    sizes = remote_sizes(cfg, files)
    local_dir = cfg.model.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    pending = {f: s for f, s in sizes.items() if not _complete(local_dir / f, s)}
    for f in files:
        state = "to download" if f in pending else "present"
        print(f"  {f}  {sizes[f] / GB:.2f} GB  ({state})")
    need = sum(pending.values())
    free = shutil.disk_usage(local_dir).free
    if need and free < need + 2 * GB:
        raise QwenError(f"Not enough disk space in {local_dir}: need {need / GB:.1f} GB, free {free / GB:.1f} GB.")

    paths = []
    for f in files:
        # Resumes partial downloads and skips files that match the remote version.
        path = hf_hub_download(cfg.model.repo, f, revision=cfg.model.revision, local_dir=local_dir)
        paths.append(Path(path))
    return paths


def _complete(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size
