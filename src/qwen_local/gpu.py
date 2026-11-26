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
    # CUDA compute capability, e.g. (8, 9) for Ada (RTX 40) and (12, 0) for Blackwell (RTX 50).
    compute_cap: tuple[int, int] | None

    @property
    def compute_cap_str(self) -> str:
        return f"{self.compute_cap[0]}.{self.compute_cap[1]}" if self.compute_cap else "unknown"


def _parse_version(text: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)", text.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


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
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    )
    gpus: list[GpuInfo] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            index, total, used, free = int(parts[0]), int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        gpus.append(GpuInfo(index, parts[1], total, used, free, parts[5], _parse_version(parts[6])))
    return gpus


def cuda_driver_version() -> tuple[int, int] | None:
    """Highest CUDA version the installed driver supports, from the nvidia-smi banner."""
    out = _run_nvidia_smi()
    # Older drivers print "CUDA Version: 12.4", newer ones "CUDA UMD Version: 13.5".
    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_builds_natively_for(cuda: tuple[int, int], compute_cap: tuple[int, int]) -> bool:
    """Whether a llama.cpp build made with this CUDA version has native code for the GPU.

    Covers the CUDA versions llama.cpp ships Windows builds for (12.x and 13.x).
    """
    if compute_cap >= (10, 0):
        return cuda >= (12, 8)  # Blackwell (RTX 50) arrived in CUDA 12.8
    if compute_cap < (7, 5):
        return cuda < (13, 0)  # CUDA 13 dropped Maxwell, Pascal and Volta
    return True
