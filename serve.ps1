# Starts llama-server with the model: web UI and OpenAI-compatible API at http://127.0.0.1:8080
# Any `qwen serve` option can be passed through, e.g. .\serve.ps1 --ctx 65536 --vision
& (Join-Path $PSScriptRoot 'qwen.ps1') serve @args
exit $LASTEXITCODE
