# qwen-local

Run [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) locally on Windows with
[llama.cpp](https://github.com/ggml-org/llama.cpp), using the
[Unsloth Dynamic IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) GGUF (14.35 GB).
You get a web chat UI, an OpenAI-compatible API, and a terminal chat client.

Everything stays inside this folder: the Python interpreter (`.python`), the virtual
environment (`.venv`), llama.cpp (`runtime`) and the model weights (`models`).

## Requirements

- Windows 10/11 x64, PowerShell 7 or Windows PowerShell 5.1
- NVIDIA GPU with 16 GB VRAM for full speed (RTX 4080/4090/5080 class) and a recent driver.
  With less free VRAM, llama.cpp keeps some layers on the CPU: it still works, only slower.
  Without an NVIDIA GPU, the CPU build is used.
- ~20 GB free disk space (model 14.35 GB, llama.cpp ~1 GB, optional vision projector 0.9 GB)
- [uv](https://docs.astral.sh/uv/); `setup.ps1` installs it when it is missing

## Quick start

```powershell
.\setup.ps1     # uv, Python, dependencies, llama.cpp, model download (resumable)
.\serve.ps1     # start the server; keep this window open
```

Open http://127.0.0.1:8080 for the web UI, or chat in a second terminal:

```powershell
.\chat.ps1
```

For image input, run `.\setup.ps1 -Vision` (downloads the 0.9 GB vision projector) and
start the server with `.\serve.ps1 --vision`.

For ~1.6x faster generation with a smaller context, see [Speed and memory](#speed-and-memory).

## Commands

`.\qwen.ps1 <command>` runs the CLI with the project's Python. Once `setup.ps1` has run,
`uv run qwen <command>` works as well.

| Command | What it does |
| --- | --- |
| `doctor` | Shows the GPU and free VRAM, installed llama.cpp, downloaded files and server status |
| `setup [--vision] [--upgrade-llama]` | `install-llama` + `download` + `doctor` |
| `install-llama [--upgrade] [--release bNNNN] [--backend cuda-13.5]` | Downloads a prebuilt llama.cpp from GitHub |
| `download [--vision]` | Downloads the model files from Hugging Face |
| `serve [--ctx N] [--vision] [--mtp] [--effort low] [--dry-run]` | Runs `llama-server` |
| `chat [-p "question"] [--no-think] [--effort low] [--hide-thinking]` | Terminal chat; `-p` asks once and exits |

`serve.ps1` and `chat.ps1` pass their arguments through, e.g. `.\serve.ps1 --ctx 65536`.
Options that `serve` does not know go to `llama-server` unchanged: `.\serve.ps1 --metrics`.
`--dry-run` prints the resulting `llama-server` command line.

In `chat`, type `/think off`, `/effort low`, `/reset` or `/exit`; `"""` starts and ends a
multi-line message.

## Configuration

Settings live in [config.toml](config.toml). Put personal overrides in `config.local.toml`
(same layout, only the keys you change; git-ignored), for example:

```toml
[server]
ctx_size = 65536
reasoning_effort = "low"
```

Main settings:

- `server.ctx_size`: context window. The model supports 262,144 tokens. Only 16 of its
  64 layers use attention (the rest are Gated DeltaNet with a fixed-size state), so the
  KV cache is small: about 34 KiB per token at `q8_0`, or 1.1 GiB for 32K tokens.
- `server.gpu_layers = "auto"`: llama.cpp fits as many layers on the GPU as free VRAM
  allows, keeping `fit_target_mib` free for other programs.
- `server.reasoning_effort`: `xhigh` (the model's default, longest thinking), `medium`,
  `low`, or `off`.
- `sampling.thinking` / `sampling.instruct`: the sampling settings the Qwen team recommends
  for each mode.
- `model.repo` / `model.file`: to try another quant, e.g. bartowski's
  `Qwen3.8-27B-IQ4_XS.gguf` (15.48 GB, which needs more VRAM).
- `llama_cpp.release`: `latest` or a pinned build tag. Update with
  `.\qwen.ps1 install-llama --upgrade`.

## Using the API

The server speaks the OpenAI Chat Completions API at `http://127.0.0.1:8080/v1`; the model
name is `qwen3.8-27b` and no API key is needed.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
    # Optional: thinking control per request.
    extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
)
print(response.choices[0].message.content)
```

The reasoning comes back separately in `message.reasoning_content`. Requests without thinking
should use the instruct sampling: `temperature=0.7, top_p=0.8, presence_penalty=1.5`.

## Speed and memory

Measured on an RTX 4080 (16 GB) with llama.cpp b12728, generating 400 tokens:

| Setup | Generation speed |
| --- | --- |
| Default: 32K context, all layers on the GPU | 40 tokens/s |
| `.\serve.ps1 --mtp --ctx 16384` | 62-66 tokens/s |
| `--mtp` with the 32K context (does not fit in VRAM) | 41-47 tokens/s |
| Other apps holding ~3.7 GB of VRAM, some layers on the CPU | 25 tokens/s |

- MTP (multi-token prediction) uses the draft layer built into the model file to guess
  the next tokens; about 60% of the guesses are accepted. It needs ~0.6 GB more VRAM,
  which on a 16 GB GPU leaves room for a 16K context only.
- `doctor` and `serve` report whether the model fits in free VRAM. Browsers, games and other
  GPU apps reduce it; when the model does not fit, generation slows down a lot.
- On Windows, set *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem Fallback* in the
  NVIDIA Control Panel (for `llama-server.exe`, or globally). Otherwise the driver may
  silently move VRAM into system RAM, which is far slower than llama.cpp's own CPU offload.
- Vision keeps the projector on the CPU by default (`mmproj_offload = false`) so it does not
  take VRAM from the model; set it to `true` if you have room.

## Layout

```
setup.ps1, serve.ps1, chat.ps1, qwen.ps1   entry points
scripts/env.ps1                            keeps uv's Python inside .python
config.toml                                settings
src/qwen_local/                            the `qwen` CLI
.python/  .venv/                           project-local Python and environment
runtime/llama.cpp/                         llama.cpp builds (current.json marks the active one)
models/                                    GGUF files
```

## Uninstall

Delete the folder. Nothing is installed outside it, apart from uv itself, the uv cache
(`uv cache clean` empties it) and a few small files huggingface_hub keeps in
`~\.cache\huggingface`.
