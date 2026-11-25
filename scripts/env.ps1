# Dot-sourced by the scripts in the project root.
# Keeps the Python interpreter inside the project: uv installs and looks up
# managed Pythons in .\.python instead of the global uv directory.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot '.python'

function Assert-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv is not installed. Run .\setup.ps1 first.'
    }
}
