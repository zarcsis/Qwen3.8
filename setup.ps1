<#
.SYNOPSIS
    Sets up everything needed to run Qwen3.8-27B locally.

.DESCRIPTION
    1. Installs uv if it is missing (winget, or the official installer).
    2. Installs the Python from .python-version into .\.python (project-local).
    3. Creates .venv and installs the Python dependencies.
    4. Downloads a prebuilt llama.cpp (CUDA build matching your driver) into .\runtime.
    5. Downloads the model into .\models.

    Safe to re-run: finished steps are skipped and interrupted downloads resume.

.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -Vision
#>
[CmdletBinding()]
param(
    # Also download the vision projector (~0.9 GB) for image input.
    [switch]$Vision,
    # Replace the installed llama.cpp with the newest build.
    [switch]$UpgradeLlama,
    # Install everything except the model weights.
    [switch]$SkipModel
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\env.ps1')

function Invoke-Step([string]$Title, [scriptblock]$Action) {
    Write-Host "`n==> $Title" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Action
    if ($LASTEXITCODE) { throw "$Title failed (exit code $LASTEXITCODE)." }
}

function Install-Uv {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id astral-sh.uv -e --silent --accept-source-agreements --accept-package-agreements
    }
    else {
        powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    }
    # Pick up the PATH entry the installer added, without restarting the shell.
    $env:Path = @(
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User')
    ) -join ';'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv was installed but is not on PATH; open a new terminal and run setup.ps1 again.'
    }
}

Push-Location $ProjectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Invoke-Step 'Installing uv' { Install-Uv }
    }
    Write-Host (uv --version)

    Invoke-Step "Installing Python $((Get-Content .python-version -TotalCount 1).Trim()) into .python" {
        # --no-bin / --no-registry: no python.exe shims in ~\.local\bin and no Windows
        # registry entry, so this interpreter stays private to the project.
        uv python install --no-bin --no-registry
    }

    Invoke-Step 'Installing Python dependencies into .venv' { uv sync }

    $qwenArgs = @('setup')
    if ($Vision) { $qwenArgs += '--vision' }
    if ($UpgradeLlama) { $qwenArgs += '--upgrade-llama' }
    if ($SkipModel) { $qwenArgs += '--skip-model' }
    Invoke-Step 'Installing llama.cpp and downloading the model' { uv run qwen @qwenArgs }

    Write-Host "`nDone. Start the server with .\serve.ps1, then open http://127.0.0.1:8080 or run .\chat.ps1" -ForegroundColor Green
}
finally {
    Pop-Location
}
