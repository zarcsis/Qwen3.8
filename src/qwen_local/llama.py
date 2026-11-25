"""Install prebuilt llama.cpp binaries from GitHub releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from qwen_local import QwenError, gpu
from qwen_local.config import Config

API = "https://api.github.com/repos/ggml-org/llama.cpp"
BUILD_TAG = re.compile(r"^b\d+$")
CURRENT_FILE = "current.json"
SERVER_EXE = "llama-server.exe"


@dataclass(frozen=True)
class Installed:
    tag: str
    backend: str
    bin_dir: Path

    @property
    def server(self) -> Path:
        return self.bin_dir / SERVER_EXE


@dataclass(frozen=True)
class Plan:
    tag: str
    backend: str
    assets: tuple[dict[str, Any], ...]


def _client() -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "qwen-local"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0))


def _get_json(client: httpx.Client, url: str, **params: Any) -> Any:
    response = client.get(url, params=params)
    if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
        raise QwenError("GitHub API rate limit reached; wait an hour or set GITHUB_TOKEN.")
    if response.status_code == 404:
        raise QwenError(f"Not found on GitHub: {url}")
    response.raise_for_status()
    return response.json()


def read_installed(cfg: Config) -> Installed | None:
    path = cfg.llama.dir / CURRENT_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    installed = Installed(data["tag"], data["backend"], cfg.llama.dir / data["bin_dir"])
    return installed if installed.server.is_file() else None


def server_executable(cfg: Config) -> Path:
    if cfg.llama.server:
        if not cfg.llama.server.is_file():
            raise QwenError(f"llama_cpp.server points to a missing file: {cfg.llama.server}")
        return cfg.llama.server
    installed = read_installed(cfg)
    if not installed:
        raise QwenError("llama.cpp is not installed; run: qwen install-llama")
    return installed.server


def _cuda_assets(release: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    pattern = re.compile(rf"^llama-{re.escape(release['tag_name'])}-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$")
    found = {}
    for asset in release["assets"]:
        if match := pattern.match(asset["name"]):
            found[(int(match.group(1)), int(match.group(2)))] = asset
    return found


def _plan(release: dict[str, Any], backend: str) -> Plan | None:
    """Pick the assets for a backend, or None if the release does not have them."""
    tag = release["tag_name"]
    by_name = {a["name"]: a for a in release["assets"]}

    if backend == "cuda" or backend.startswith("cuda-"):
        builds = _cuda_assets(release)
        if backend == "cuda":
            driver = gpu.cuda_driver_version()
            usable = [v for v in builds if driver is None or v <= driver]
            if not usable:
                return None
            version = max(usable)
        else:
            try:
                major, minor = backend.removeprefix("cuda-").split(".")
                version = (int(major), int(minor))
            except ValueError:
                raise QwenError(f"Bad llama_cpp.backend '{backend}'; expected e.g. cuda-13.5") from None
            if version not in builds:
                return None
        runtime = by_name.get(f"cudart-llama-bin-win-cuda-{version[0]}.{version[1]}-x64.zip")
        if runtime is None:
            return None
        return Plan(tag, f"cuda-{version[0]}.{version[1]}", (builds[version], runtime))

    asset = by_name.get(f"llama-{tag}-bin-win-{backend}-x64.zip")
    return Plan(tag, backend, (asset,)) if asset else None


def _resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if gpu.cuda_driver_version() else "cpu"


def find_plan(release: str, backend: str) -> Plan:
    backend = _resolve_backend(backend)
    with _client() as client:
        if release != "latest":
            plan = _plan(_get_json(client, f"{API}/releases/tags/{release}"), backend)
            if plan is None:
                raise QwenError(f"llama.cpp {release} has no Windows x64 build for backend '{backend}'.")
            return plan
        # The "latest" release is a semver tag without binaries; the bNNNN builds are prereleases.
        for page in (1, 2, 3):
            for rel in _get_json(client, f"{API}/releases", per_page=30, page=page):
                if BUILD_TAG.match(rel["tag_name"]) and not rel["draft"]:
                    plan = _plan(rel, backend)
                    if plan is not None:
                        return plan
    raise QwenError(f"No recent llama.cpp release has a Windows x64 build for backend '{backend}'.")


def _download(client: httpx.Client, asset: dict[str, Any], dest: Path) -> None:
    expected = None
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        expected = digest.removeprefix("sha256:")

    already = dest.is_file() and dest.stat().st_size == asset["size"]
    if already and (expected is None or _sha256(dest) == expected):
        print(f"  {asset['name']}: already downloaded")
        return

    part = dest.with_name(dest.name + ".part")
    hasher = hashlib.sha256()
    with client.stream("GET", asset["browser_download_url"]) as response:
        response.raise_for_status()
        with (
            part.open("wb") as f,
            tqdm(total=asset["size"], unit="B", unit_scale=True, unit_divisor=1024, desc=f"  {asset['name']}") as bar,
        ):
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
                hasher.update(chunk)
                bar.update(len(chunk))
    if expected and hasher.hexdigest() != expected:
        part.unlink()
        raise QwenError(f"Checksum mismatch for {asset['name']}; try again.")
    part.replace(dest)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _retry[T](action: Callable[[], T], attempts: int = 10) -> T:
    """Retry a file operation that antivirus scanners briefly block on Windows."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def _extract(archives: list[Path], target: Path) -> Path:
    # Extract in place rather than into a staging directory: renaming a directory full of
    # freshly written DLLs fails while Windows Defender scans them. current.json, written
    # after this succeeds, marks the install as complete.
    if target.exists():
        _retry(lambda: shutil.rmtree(target))
    target.mkdir(parents=True)
    root = target.resolve()
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                # Refuse entries that would escape the target directory.
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise QwenError(f"Unsafe path in {archive.name}: {member.filename}")
            zf.extractall(target)
    servers = sorted(target.rglob(SERVER_EXE), key=lambda p: len(p.parts))
    if not servers:
        raise QwenError(f"{SERVER_EXE} not found in the downloaded archive.")
    server_dir = servers[0].parent
    if server_dir != target:
        # The CUDA runtime archive is flat; keep its DLLs next to the executables.
        for item in target.iterdir():
            if item.is_file():
                _retry(lambda item=item: shutil.move(item, server_dir / item.name))
    return server_dir


def install(cfg: Config, *, upgrade: bool = False, release: str | None = None, backend: str | None = None) -> Installed:
    if sys.platform != "win32":
        raise QwenError("install-llama downloads Windows x64 builds only; point llama_cpp.server at your llama-server.")

    installed = read_installed(cfg)
    if installed and not (upgrade or release or backend):
        print(f"llama.cpp {installed.tag} ({installed.backend}) is installed: {installed.bin_dir}")
        return installed

    plan = find_plan(release or cfg.llama.release, backend or cfg.llama.backend)
    print(f"llama.cpp {plan.tag}, backend {plan.backend}")
    if installed and (installed.tag, installed.backend) == (plan.tag, plan.backend):
        print("  already the installed version")
        return installed

    target = cfg.llama.dir / f"{plan.tag}-{plan.backend}"
    downloads = cfg.llama.dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        archives = []
        for asset in plan.assets:
            dest = downloads / asset["name"]
            _download(client, asset, dest)
            archives.append(dest)
    bin_dir = _extract(archives, target)

    current = {"tag": plan.tag, "backend": plan.backend, "bin_dir": bin_dir.relative_to(cfg.llama.dir).as_posix()}
    (cfg.llama.dir / CURRENT_FILE).write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    for archive in archives:
        archive.unlink()
    if installed and installed.bin_dir.exists():
        # Keep only the active build; the previous one is ~1 GB of DLLs.
        old_root = cfg.llama.dir / installed.bin_dir.relative_to(cfg.llama.dir).parts[0]
        shutil.rmtree(old_root, ignore_errors=True)

    result = Installed(plan.tag, plan.backend, bin_dir)
    version = subprocess.run([result.server, "--version"], capture_output=True, text=True, check=False)
    print("  " + (version.stdout + version.stderr).strip().replace("\n", "\n  "))
    return result
