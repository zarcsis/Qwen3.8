# qwen-local

Run [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) locally on Windows with
[llama.cpp](https://github.com/ggml-org/llama.cpp), using the
[Unsloth Dynamic IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) GGUF (14.35 GB).
You get a web chat UI, an OpenAI-compatible API, and a terminal chat client.

Everything stays inside this folder: the Python interpreter (`.python`), the virtual
environment (`.venv`), llama.cpp (`runtime`) and the model weights (`models`).

## Requirements

- Windows 10/11 x64, PowerShell 7 or Windows PowerShell 5.1
- NVIDIA GPU with 16 GB VRAM or more for full speed and a recent driver. Tested on an
  RTX 4080 and an RTX 5070 Ti; nothing in the setup is specific to one card (see
  [Changing the GPU](#changing-the-gpu)). With less free VRAM, llama.cpp keeps some layers
  on the CPU: it still works, only slower. Without an NVIDIA GPU, the CPU build is used.
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

For ~1.5x faster generation with a smaller context, see [Speed and memory](#speed-and-memory).

## Commands

`.\qwen.ps1 <command>` runs the CLI with the project's Python. Once `setup.ps1` has run,
`uv run qwen <command>` works as well.

| Command | What it does |
| --- | --- |
| `doctor` | Shows the GPU, free VRAM and what fits in it, installed llama.cpp, files and server status |
| `setup [--vision] [--upgrade-llama]` | `install-llama` + `download` + `doctor` |
| `install-llama [--upgrade] [--release bNNNN] [--backend cuda-13.5]` | Downloads a prebuilt llama.cpp from GitHub |
| `download [--vision]` | Downloads the model files from Hugging Face |
| `serve [--ctx N] [--vision] [--mtp] [--effort low] [--dry-run]` | Runs `llama-server` |
| `chat [-p "question"] [--no-think] [--effort low] [--hide-thinking]` | Terminal chat; `-p` asks once and exits |
| `bench [--profile default\|mtp] [--runs 3] [--ctx N] [--prompt "..."]` | Starts its own server per profile and measures prompt and generation speed |

`serve.ps1` and `chat.ps1` pass their arguments through, e.g. `.\serve.ps1 --ctx 16384`.
Options that `serve` does not know go to `llama-server` unchanged: `.\serve.ps1 --metrics`.
`--dry-run` prints the resulting `llama-server` command line.

In `chat`, type `/think off`, `/effort low`, `/reset` or `/exit`; `"""` starts and ends a
multi-line message.

## Configuration

Settings live in [config.toml](config.toml). Put personal overrides in `config.local.toml`
(same layout, only the keys you change; git-ignored), for example:

```toml
[server]
mtp = true
reasoning_effort = "low"
```

Main settings:

- `server.ctx_size = 0`: llama.cpp picks the largest context window that fits in the free
  VRAM with every layer on the GPU, keeping `fit_target_mib` (1 GiB) free for other
  programs. On a 16 GB card that is 20-26K tokens, depending on what else uses VRAM; with
  more VRAM it grows, up to the model's 262,144 tokens. Only 16 of the model's 64 layers use
  attention (the rest are Gated DeltaNet with a fixed-size state), so the KV cache is small:
  about 34 KiB per token at `q8_0`. A fixed value is possible, but see
  [Speed and memory](#speed-and-memory) for what happens when it does not fit.
- `server.gpu_layers = "auto"`: llama.cpp fits as many layers on the GPU as free VRAM
  allows; with an automatic context, it shrinks the context first.
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

Generation speed in tokens/s with llama.cpp b12728: the same prompt, 400 tokens without
thinking, median of 3 runs (`.\qwen.ps1 bench` repeats the measurement on your machine):

| Setup | RTX 4080 | RTX 5070 Ti |
| --- | --- | --- |
| Current defaults: automatic context (26K tokens), all layers on the GPU | - | 47 |
| Current defaults with `--mtp` (automatic context 13.5K) | - | 70 |
| Previous defaults: fixed 32K context, 512 MiB margin | 40 | 40, often 20-33 |
| Previous defaults with MTP and a 16K context | 62-66 | 65-66 |
| Decoding without the server (`llama-bench`, tg128) | - | 50.5 |
| Other apps holding ~3.7 GB of VRAM, some layers on the CPU | 25 | - |

The RTX 4080 numbers were measured before it was replaced, so it has no results for the
current defaults. On the RTX 5070 Ti, prompt processing runs at 1,350-1,500 tokens/s
(3,000-token prompt) and 2,150 tokens/s in `llama-bench` (pp512).

- Generation speed is set by memory bandwidth: each token reads all 13 GB of weights. The
  RTX 5070 Ti has 896 GB/s against the RTX 4080's 717 GB/s, which puts it ~25% ahead when
  the model fits comfortably (50.5 tokens/s in `llama-bench`; the same ratio gives ~40 for
  the RTX 4080). Sampling in the server costs another ~3 tokens/s.
- With the old fixed 32K context, the model, KV cache and buffers left only ~300 MiB of VRAM.
  Windows then moved part of the server's memory into system RAM, which made generation
  slow and erratic and hid the RTX 5070 Ti's advantage. The automatic context with a 1 GiB
  margin avoids that; `doctor` and `serve` print the context and layers that fit.
- MTP (multi-token prediction) uses the draft layer built into the model file to guess the
  next tokens; about 58% of the guesses are accepted. It takes ~0.5 GiB of VRAM, which the
  server subtracts from the automatic context, so the context roughly halves.
- Browsers, games and other GPU apps reduce the free VRAM and therefore the context. When
  even a 4K context does not fit, layers move to the CPU and generation slows down a lot.
- On Windows, set *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem Fallback* in the
  NVIDIA Control Panel (for `llama-server.exe`, or globally). Otherwise the driver may
  silently move VRAM into system RAM, which is far slower than llama.cpp's own CPU offload.
- Vision keeps the projector on the CPU by default (`mmproj_offload = false`) so it does not
  take VRAM from the model; set it to `true` if you have room.

## Changing the GPU

Nothing in the configuration is tied to one card. After swapping the GPU or updating the
driver, run `.\setup.ps1` again:

- It checks the installed llama.cpp build against the GPU and driver, and replaces it when
  they no longer match. `doctor` and `serve` warn about a mismatch too.
- `install-llama` picks the newest CUDA build that the driver supports and that has native
  code for every GPU present. RTX 50 cards (Blackwell, compute capability 12.0) need a build
  made with CUDA 12.8 or newer, which means the CUDA 13 build. GPUs older than the RTX 20
  series (compute capability below 7.5) need the CUDA 12 build.
- The automatic context adapts to the new card's VRAM.

`.\qwen.ps1 bench` then measures the new card; each run is appended to
`runtime\bench\results.jsonl`.

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
