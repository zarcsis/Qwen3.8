"""Minimal reader for GGUF metadata (the key-value header of a model file)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

# GGUF value types: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


def _read(f: BinaryIO, fmt: str) -> int | float | bool:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))[0]


def _read_string(f: BinaryIO) -> str:
    length = int(_read(f, "Q"))
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, kind: int) -> object:
    if kind in _SCALARS:
        return _read(f, _SCALARS[kind])
    if kind == _STRING:
        return _read_string(f)
    if kind == _ARRAY:
        item_kind = int(_read(f, "I"))
        count = int(_read(f, "Q"))
        if item_kind in _SCALARS:
            # Skip numeric arrays without decoding them.
            f.seek(struct.calcsize("<" + _SCALARS[item_kind]) * count, 1)
        else:
            for _ in range(count):
                _read_value(f, item_kind)
        return None
    raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: Path, wanted: set[str]) -> dict[str, object]:
    """Return the metadata values for the keys in `wanted` ("{arch}" expands to the architecture)."""
    found: dict[str, object] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path.name} is not a GGUF file")
        _read(f, "I")  # version
        _read(f, "Q")  # tensor count
        kv_count = int(_read(f, "Q"))
        arch = None
        for _ in range(kv_count):
            key = _read_string(f)
            value = _read_value(f, int(_read(f, "I")))
            if key == "general.architecture":
                arch = str(value)
            for w in wanted:
                if key == (w.replace("{arch}", arch) if arch else w):
                    found[w] = value
            if len(found) == len(wanted):
                break
    return found


def gpu_layer_count(path: Path) -> int | None:
    """How many layers llama.cpp can offload (its -ngl value for "all"), or None if unknown.

    That is the repeating blocks plus the output layer; MTP (nextn) blocks are counted
    in block_count but are not part of the main model's layers.
    """
    keys = {"{arch}.block_count", "{arch}.nextn_predict_layers"}
    try:
        meta = read_metadata(path, keys)
    except OSError, ValueError, struct.error:
        return None
    blocks = meta.get("{arch}.block_count")
    nextn = meta.get("{arch}.nextn_predict_layers", 0)
    if not isinstance(blocks, int) or not isinstance(nextn, int):
        return None
    return blocks - nextn + 1
