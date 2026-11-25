"""NVIDIA GPU detection through nvidia-smi."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    driver: str


def _run_nvidia_smi(*args: str) -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run([exe, *args], capture_output=True, text=True, timeout=30, check=False)
    except OSError, subprocess.TimeoutExpired:
        return None
    return result.stdout if result.returncode == 0 else None


def query_gpus() -> list[GpuInfo]:
    out = _run_nvidia_smi(
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    )
    gpus: list[GpuInfo] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(GpuInfo(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), int(parts[4]), parts[5]))
        except ValueError:
            continue
    return gpus


def cuda_driver_version() -> tuple[int, int] | None:
    """Highest CUDA version the installed driver supports, from the nvidia-smi banner."""
    out = _run_nvidia_smi()
    # Older drivers print "CUDA Version: 12.4", newer ones "CUDA UMD Version: 13.5".
    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None
