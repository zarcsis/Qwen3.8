"""Project configuration: config.toml, optionally overlaid by config.local.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_local import QwenError

CONFIG_FILE = "config.toml"
LOCAL_CONFIG_FILE = "config.local.toml"
# Levels the Qwen3.8 chat template accepts (any other value makes it raise),
# plus "off", which disables thinking altogether.
REASONING_EFFORTS = ("xhigh", "medium", "low", "off")


def project_root() -> Path:
    """Locate the project directory (the one that holds config.toml)."""
    env = os.environ.get("QWEN_LOCAL_ROOT")
    if env:
        return Path(env).resolve()
    # Editable install: src/qwen_local/config.py -> project root is two levels up.
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / CONFIG_FILE).is_file():
        return candidate
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / CONFIG_FILE).is_file():
            return directory
    raise QwenError(f"Cannot find {CONFIG_FILE}; run from the project directory or set QWEN_LOCAL_ROOT.")


@dataclass(frozen=True)
class ModelConfig:
    repo: str
    revision: str
    file: str
    mmproj_file: str
    dir: Path
    alias: str

    @property
    def local_dir(self) -> Path:
        return self.dir / self.repo

    @property
    def path(self) -> Path:
        return self.local_dir / self.file

    @property
    def mmproj_path(self) -> Path:
        return self.local_dir / self.mmproj_file


@dataclass(frozen=True)
class LlamaConfig:
    release: str
    backend: str
    dir: Path
    server: Path | None


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repeat_penalty: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    ctx_size: int
    gpu_layers: str
    fit_target_mib: int
    cache_type_k: str
    cache_type_v: str
    flash_attn: str
    parallel: int
    batch_size: int
    ubatch_size: int
    threads: int
    vision: bool
    mmproj_offload: bool
    mtp: bool
    mtp_draft_max: int
    reasoning_effort: str
    extra_args: tuple[str, ...]

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True)
class Config:
    root: Path
    model: ModelConfig
    llama: LlamaConfig
    server: ServerConfig
    thinking: Sampling
    instruct: Sampling


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise QwenError(f"{path}: {e}") from e


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise QwenError(f"{CONFIG_FILE}: missing [{name}] section")
    return value


def _get(section: dict[str, Any], key: str, kind: type, where: str) -> Any:
    if key not in section:
        raise QwenError(f"{CONFIG_FILE}: missing '{key}' in [{where}]")
    value = section[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise QwenError(f"{CONFIG_FILE}: '{key}' in [{where}] must be {kind.__name__}, got {value!r}")
    return value


def _sampling(data: dict[str, Any], name: str) -> Sampling:
    s = _section(_section(data, "sampling"), name)
    where = f"sampling.{name}"
    return Sampling(
        temperature=_get(s, "temperature", float, where),
        top_p=_get(s, "top_p", float, where),
        top_k=_get(s, "top_k", int, where),
        min_p=_get(s, "min_p", float, where),
        presence_penalty=_get(s, "presence_penalty", float, where),
        repeat_penalty=_get(s, "repeat_penalty", float, where),
    )


def load(path: Path | None = None) -> Config:
    root = project_root()
    path = path or root / CONFIG_FILE
    data = _read_toml(path)
    local = path.with_name(LOCAL_CONFIG_FILE)
    if local.is_file():
        data = _merge(data, _read_toml(local))

    def resolve(p: str) -> Path:
        candidate = Path(os.path.expandvars(p)).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    m = _section(data, "model")
    model = ModelConfig(
        repo=_get(m, "repo", str, "model"),
        revision=_get(m, "revision", str, "model"),
        file=_get(m, "file", str, "model"),
        mmproj_file=_get(m, "mmproj_file", str, "model"),
        dir=resolve(_get(m, "dir", str, "model")),
        alias=_get(m, "alias", str, "model"),
    )

    lc = _section(data, "llama_cpp")
    server_override = _get(lc, "server", str, "llama_cpp")
    llama = LlamaConfig(
        release=_get(lc, "release", str, "llama_cpp"),
        backend=_get(lc, "backend", str, "llama_cpp"),
        dir=resolve(_get(lc, "dir", str, "llama_cpp")),
        server=resolve(server_override) if server_override else None,
    )

    s = _section(data, "server")
    gpu_layers = s.get("gpu_layers", "auto")
    effort = _get(s, "reasoning_effort", str, "server")
    if effort not in REASONING_EFFORTS:
        raise QwenError(f"{CONFIG_FILE}: reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
    server = ServerConfig(
        host=_get(s, "host", str, "server"),
        port=_get(s, "port", int, "server"),
        ctx_size=_get(s, "ctx_size", int, "server"),
        gpu_layers=str(gpu_layers),
        fit_target_mib=_get(s, "fit_target_mib", int, "server"),
        cache_type_k=_get(s, "cache_type_k", str, "server"),
        cache_type_v=_get(s, "cache_type_v", str, "server"),
        flash_attn=_get(s, "flash_attn", str, "server"),
        parallel=_get(s, "parallel", int, "server"),
        batch_size=_get(s, "batch_size", int, "server"),
        ubatch_size=_get(s, "ubatch_size", int, "server"),
        threads=_get(s, "threads", int, "server"),
        vision=_get(s, "vision", bool, "server"),
        mmproj_offload=_get(s, "mmproj_offload", bool, "server"),
        mtp=_get(s, "mtp", bool, "server"),
        mtp_draft_max=_get(s, "mtp_draft_max", int, "server"),
        reasoning_effort=effort,
        extra_args=tuple(str(a) for a in s.get("extra_args", [])),
    )

    return Config(
        root=root,
        model=model,
        llama=llama,
        server=server,
        thinking=_sampling(data, "thinking"),
        instruct=_sampling(data, "instruct"),
    )
