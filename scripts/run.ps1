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

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "Run scripts\setup.ps1 first." }

# Make Python print UTF-8 to the console (Chinese messages)
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$cliArgs = @()
if ($ListDevices) { $cliArgs += "--list-devices" }
if ($Settings) { $cliArgs += "--settings" }
if ($Rest) { $cliArgs += $Rest }
& $VenvPy -m livesub @cliArgs
exit $LASTEXITCODE
