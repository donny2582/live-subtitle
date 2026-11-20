<#
.SYNOPSIS
    Build a portable folder dist\LiveSub\ containing LiveSub.exe (PyInstaller, one-dir).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -Console   # keep a console window for debugging
.NOTES
    One-dir (not one-file) is intentional: with CUDA DLLs a one-file exe would be
    several GB and unpack to %TEMP% on every launch (slow start, antivirus noise).
    Whisper models are NOT bundled; they download to dist\LiveSub\models on first run.
#>
param([switch]$Console)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "Run scripts\setup.ps1 first." }

& $VenvPy -m pip install "pyinstaller>=6.10"
if ($LASTEXITCODE -ne 0) { throw "pyinstaller install failed" }

if ($Console) { $env:LIVESUB_CONSOLE = "1" } else { Remove-Item Env:LIVESUB_CONSOLE -ErrorAction SilentlyContinue }
& $VenvPy -m PyInstaller packaging\livesub.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$Dist = Join-Path $Root "dist\LiveSub"
foreach ($f in @("config.example.toml", "glossary.example.csv", "README.md")) {
    Copy-Item $f $Dist -Force
}
Write-Host ""
Write-Host "Done: $Dist\LiveSub.exe" -ForegroundColor Green
Write-Host "Zip the whole LiveSub folder to move it to another PC."
