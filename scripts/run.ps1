<#
.SYNOPSIS
    Start the live subtitle overlay.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -ListDevices
    powershell -ExecutionPolicy Bypass -File scripts\run.ps1 --model small --log-level DEBUG
#>
param(
    [switch]$ListDevices,
    [switch]$Settings,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"   # stop on the first error instead of continuing
$Root = Split-Path -Parent $PSScriptRoot   # project root = parent of the scripts folder
Set-Location $Root
# Always use the project's own Python in .venv (created by setup.ps1)
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "Run scripts\setup.ps1 first." }

# Make Python print UTF-8 to the console (Chinese messages)
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Translate PowerShell-style switches into the Python program's --flags;
# anything else (e.g. --engine claude) is passed through unchanged.
$cliArgs = @()
if ($ListDevices) { $cliArgs += "--list-devices" }
if ($Settings) { $cliArgs += "--settings" }
if ($Rest) { $cliArgs += $Rest }
# Runs src\livesub\__main__.py -> app.main()
& $VenvPy -m livesub @cliArgs
exit $LASTEXITCODE
