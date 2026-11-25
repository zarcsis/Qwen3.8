# Runs the qwen CLI with the project-local Python. Examples:
#   .\qwen.ps1 doctor
#   .\qwen.ps1 serve --ctx 16384
#   .\qwen.ps1 chat --no-think
. (Join-Path $PSScriptRoot 'scripts\env.ps1')
Assert-Uv
uv run --project $PSScriptRoot qwen @args
exit $LASTEXITCODE
