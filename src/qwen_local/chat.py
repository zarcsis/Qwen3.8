"""Terminal chat client for the llama-server OpenAI-compatible API."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from qwen_local import QwenError
from qwen_local.config import Config

HELP = """Commands:
  /think on|off            turn thinking on or off
  /effort xhigh|medium|low thinking depth
  /reset                   start a new conversation
  /exit                    quit (also Ctrl+C or Ctrl+Z, Enter)
  \"\"\"                      start or end a multi-line message"""

EFFORTS = ("xhigh", "medium", "low")


@dataclass
class ChatSettings:
    think: bool = True
    effort: str | None = None  # None keeps the server default
    show_thinking: bool = True
    system: str | None = None
    max_tokens: int | None = None


@dataclass
class Style:
    enabled: bool

    def wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self.wrap("2", text)

    def prompt(self, text: str) -> str:
        return self.wrap("1;36", text)


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    timings: dict[str, Any] = field(default_factory=dict)


def _enable_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def wait_until_ready(client: httpx.Client, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = client.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            state = "loading the model"
        except httpx.TransportError:
            state = "not running"
        if time.monotonic() >= deadline:
            hint = "start it with .\\serve.ps1" if state == "not running" else "wait, or use --wait"
            raise QwenError(f"Server at {base_url} is {state}; {hint}.")
        time.sleep(1)


def _request_body(cfg: Config, settings: ChatSettings, messages: list[dict[str, Any]]) -> dict[str, Any]:
    preset = cfg.thinking if settings.think else cfg.instruct
    template_kwargs: dict[str, Any] = {"enable_thinking": settings.think}
    if settings.think and settings.effort:
        template_kwargs["reasoning_effort"] = settings.effort
    body: dict[str, Any] = {
        "model": cfg.model.alias,
        "messages": messages,
        "stream": True,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "min_p": preset.min_p,
        "presence_penalty": preset.presence_penalty,
        "repeat_penalty": preset.repeat_penalty,
        "chat_template_kwargs": template_kwargs,
    }
    if settings.max_tokens:
        body["max_tokens"] = settings.max_tokens
    return body


def complete(client: httpx.Client, base_url: str, body: dict[str, Any], settings: ChatSettings, style: Style) -> Reply:
    """Stream one completion to stdout and return it."""
    reply = Reply()
    in_reasoning = False
    started = time.monotonic()
    with client.stream("POST", f"{base_url}/v1/chat/completions", json=body, timeout=None) as response:
        if response.status_code != 200:
            response.read()
            raise QwenError(f"Server error {response.status_code}: {response.text}")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise QwenError(f"Server error: {chunk['error']}")
            if timings := chunk.get("timings"):
                reply.timings = timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if text := delta.get("reasoning_content"):
                    reply.reasoning += text
                    if not in_reasoning:
                        marker = "[thinking] " if settings.show_thinking else "[thinking...]"
                        print(style.dim(marker), end="", flush=True)
                        in_reasoning = True
                    if settings.show_thinking:
                        print(style.dim(text), end="", flush=True)
                if text := delta.get("content"):
                    if in_reasoning:
                        elapsed = time.monotonic() - started
                        print("\n" + style.dim(f"[thought for {elapsed:.0f}s]") + "\n")
                        in_reasoning = False
                    if not reply.content:
                        text = text.lstrip()
                    reply.content += text
                    print(text, end="", flush=True)
    print()
    t = reply.timings
    if t.get("predicted_per_second"):
        print(
            style.dim(
                f"[{t.get('prompt_n', 0)} prompt tokens at {t.get('prompt_per_second', 0):.0f} t/s, "
                f"{t.get('predicted_n', 0)} generated at {t['predicted_per_second']:.1f} t/s]"
            )
        )
    return reply


def _read_message(style: Style) -> str | None:
    """Read one user message; None means quit."""
    try:
        line = input(style.prompt("you> "))
    except EOFError, KeyboardInterrupt:
        print()
        return None
    if line.strip() != '"""':
        return line
    lines = []
    while True:
        try:
            line = input()
        except EOFError, KeyboardInterrupt:
            print()
            return None
        if line.strip() == '"""':
            return "\n".join(lines)
        lines.append(line)


def _handle_command(text: str, settings: ChatSettings, messages: list[dict[str, Any]]) -> bool:
    """Apply a /command; returns False to quit."""
    name, _, arg = text[1:].partition(" ")
    arg = arg.strip().lower()
    if name in ("exit", "quit", "q"):
        return False
    if name == "reset":
        del messages[1 if settings.system else 0 :]
        print("New conversation.")
    elif name == "think" and arg in ("on", "off"):
        settings.think = arg == "on"
        print(f"Thinking {arg}.")
    elif name == "effort" and arg in EFFORTS:
        settings.effort = arg
        settings.think = True
        print(f"Reasoning effort {arg}.")
    else:
        print(HELP)
    return True


def run(cfg: Config, base_url: str, settings: ChatSettings, *, prompt: str | None = None, wait: float = 0) -> int:
    style = Style(_enable_ansi())
    messages: list[dict[str, Any]] = []
    if settings.system:
        messages.append({"role": "system", "content": settings.system})

    with httpx.Client(timeout=httpx.Timeout(30.0, read=None)) as client:
        wait_until_ready(client, base_url, wait)

        if prompt is not None:
            if prompt == "-":
                prompt = sys.stdin.read()
            messages.append({"role": "user", "content": prompt})
            complete(client, base_url, _request_body(cfg, settings, messages), settings, style)
            return 0

        mode = f"thinking {'on' if settings.think else 'off'}"
        if settings.think and settings.effort:
            mode += f", effort {settings.effort}"
        print(style.dim(f"Chatting with {cfg.model.alias} at {base_url} ({mode}). /help for commands."))
        while True:
            text = _read_message(style)
            if text is None:
                return 0
            if not text.strip():
                continue
            if text.startswith("/"):
                if not _handle_command(text.strip(), settings, messages):
                    return 0
                continue
            messages.append({"role": "user", "content": text})
            try:
                reply = complete(client, base_url, _request_body(cfg, settings, messages), settings, style)
            except KeyboardInterrupt:
                # Closing the stream makes the server stop generating.
                print(style.dim("\n[interrupted]"))
                messages.pop()
                continue
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.reasoning:
                # The Qwen3.8 template keeps earlier reasoning in context (preserved thinking).
                assistant["reasoning_content"] = reply.reasoning
            messages.append(assistant)
