<#
.SYNOPSIS
    Create .venv and install dependencies (Windows PowerShell 5.1 / PowerShell 7).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1              # cloud ASR + cloud translation
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -LocalAsr    # + local faster-whisper (CPU)
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu         # + local faster-whisper on NVIDIA GPU
.NOTES
    Messages are ASCII on purpose: Windows PowerShell 5.1 reads BOM-less
    scripts using the system code page (Big5 on zh-TW), which garbles CJK text.
#>
param(
    [switch]$LocalAsr,     # also install faster-whisper (asr.engine = "local")
    [switch]$Gpu,          # local ASR + CUDA 12 / cuDNN 9 runtime DLLs (implies -LocalAsr)
    [switch]$Dev,          # also install pytest, ruff, mypy, pyinstaller
    [string]$Python = ""   # explicit python.exe path (optional)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Find-Python {
    if ($Python) { return ,@($Python) }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.13", "3.14", "3.11")) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return ,@("python") }
    throw "Python 3.11-3.14 not found. Install from https://www.python.org/downloads/ (check 'Add to PATH')."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $py = Find-Python
    $exe = $py[0]
    $pyArgs = @()
    if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
    & $exe @pyArgs -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,14) else 1)"
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11-3.14 is required." }
    Write-Host "Creating virtual environment (.venv) ..." -ForegroundColor Cyan
    & $exe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
if ($Gpu) {
    Write-Host "Installing local ASR + CUDA runtime DLLs ..." -ForegroundColor Cyan
    & $VenvPy -m pip install -r requirements-gpu.txt
    if ($LASTEXITCODE -ne 0) { throw "GPU packages install failed" }
} elseif ($LocalAsr) {
    Write-Host "Installing local ASR (faster-whisper) ..." -ForegroundColor Cyan
    & $VenvPy -m pip install -r requirements-local-asr.txt
    if ($LASTEXITCODE -ne 0) { throw "local ASR packages install failed" }
}
if ($Dev) {
    & $VenvPy -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "dev packages install failed" }
}
& $VenvPy -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "editable install failed" }

foreach ($pair in @(@("config.example.toml", "config.toml"), @("glossary.example.csv", "glossary.csv"))) {
    if (-not (Test-Path $pair[1])) { Copy-Item $pair[0] $pair[1] }
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "1) Set API keys once (new terminals pick them up):"
Write-Host '     setx OPENAI_API_KEY "sk-..."          (cloud ASR + translate.engine = openai)'
Write-Host '     setx ANTHROPIC_API_KEY "sk-ant-..."   (translate.engine = claude)'
Write-Host "2) Start:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1"
