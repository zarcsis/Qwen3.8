# Terminal chat with the running server (start it first with .\serve.ps1).
# Any `qwen chat` option can be passed through, e.g. .\chat.ps1 --no-think
& (Join-Path $PSScriptRoot 'qwen.ps1') chat @args
exit $LASTEXITCODE
