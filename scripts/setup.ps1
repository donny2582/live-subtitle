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
    [string]$Python = "",  # explicit python.exe path (optional)
    [switch]$NoShortcut    # do not create the desktop shortcut
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Find a usable Python: the -Python argument first, then the "py" launcher
# (trying 3.12, 3.13, 3.14, 3.11 in that order), then plain "python".
# Returns the command as an array, e.g. @("py", "-3.12") or @("python").
function Find-Python {
    if ($Python) { return ,@($Python) }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        # When a version is not installed, "py" prints "No suitable Python runtime found"
        # to stderr. Windows PowerShell 5.1 turns that into a terminating error under
        # $ErrorActionPreference = "Stop", so relax it inside this function only
        # (the change is local to the function scope) and rely on the exit code instead.
        $ErrorActionPreference = "Continue"
        foreach ($v in @("3.12", "3.13", "3.14", "3.11")) {
            try {
                & py "-$v" -c "import sys" 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
            } catch {
                # this version is not installed; try the next one
            }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return ,@("python") }
    throw "Python 3.11-3.14 not found. Install from https://www.python.org/downloads/ (check 'Add to PATH')."
}

# Step 1: create the .venv virtual environment (only the first time)
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

# Step 2: install packages into .venv (never into the system Python)
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r requirements.txt   # core: audio, UI, cloud ASR + translation
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
# Step 3: install this project itself in "editable" mode, so code changes
# under src\ take effect immediately without reinstalling
& $VenvPy -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw "editable install failed" }

# Step 4: create the user's own config.toml / glossary.csv from the examples
foreach ($pair in @(@("config.example.toml", "config.toml"), @("glossary.example.csv", "glossary.csv"))) {
    if (-not (Test-Path $pair[1])) { Copy-Item $pair[0] $pair[1] }
}

# Step 5: desktop shortcut (skip with -NoShortcut). A failure here is not fatal.
if (-not $NoShortcut) {
    try {
        & (Join-Path $PSScriptRoot "create_shortcut.ps1")
    } catch {
        Write-Host "Could not create the desktop shortcut: $_" -ForegroundColor Yellow
        Write-Host "You can retry later: powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1"
    }
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Double-click the desktop shortcut to start. The settings window opens first:"
Write-Host "  paste your API keys in the 'API keys' tab, then press the Start button."
Write-Host "(Or from a terminal: powershell -ExecutionPolicy Bypass -File scripts\run.ps1)"
