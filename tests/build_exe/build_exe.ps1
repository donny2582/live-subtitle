<#
.SYNOPSIS
    Build a portable folder dist\LiveSub\ containing LiveSub.exe (PyInstaller, one-dir),
    plus dist\LiveSub.zip ready to copy to another PC.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tests\build_exe\build_exe.ps1
    powershell -ExecutionPolicy Bypass -File tests\build_exe\build_exe.ps1 -Console   # keep a console window for debugging
    powershell -ExecutionPolicy Bypass -File tests\build_exe\build_exe.ps1 -NoZip
.NOTES
    The target PC does NOT need Python: Python itself and every package are bundled.
    Only this build machine needs the project's .venv (run scripts\setup.ps1 first).

    Your own config.toml, glossary.csv, logs and transcripts are NOT packaged; the
    example files are copied instead, and the app creates config.toml on first run.
    API keys are never packaged (they live in Windows user environment variables).

    One-dir (not one-file) is intentional: with CUDA DLLs a one-file exe would be
    several GB and unpack to %TEMP% on every launch (slow start, antivirus noise).
    Whisper models are NOT bundled; they download to dist\LiveSub\models on first run.

    Messages are ASCII on purpose: Windows PowerShell 5.1 reads BOM-less
    scripts using the system code page (Big5 on zh-TW), which garbles CJK text.
#>
param(
    [switch]$Console,   # build with a console window (shows log output; for debugging)
    [switch]$NoZip      # skip creating dist\LiveSub.zip
)

$ErrorActionPreference = "Stop"
# This script lives in <project>\tests\build_exe, so the project root is two levels up
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { throw "Run scripts\setup.ps1 first." }

# Step 1: make sure PyInstaller is installed in .venv
& $VenvPy -m pip install --quiet "pyinstaller>=6.10"
if ($LASTEXITCODE -ne 0) { throw "pyinstaller install failed" }

# Step 2: draw the app icon into build\livesub.ico (the spec embeds it into LiveSub.exe)
New-Item -ItemType Directory -Force (Join-Path $Root "build") | Out-Null
& $VenvPy -c "import sys; from livesub.ui.icon import write_ico; write_ico(sys.argv[1])" (Join-Path $Root "build\livesub.ico")
if ($LASTEXITCODE -ne 0) { throw "Failed to create the icon file" }

# Step 3: run PyInstaller
# livesub.spec reads LIVESUB_CONSOLE to decide whether the exe opens a console window
if ($Console) { $env:LIVESUB_CONSOLE = "1" } else { Remove-Item Env:LIVESUB_CONSOLE -ErrorAction SilentlyContinue }
$Spec = Join-Path $PSScriptRoot "livesub.spec"
& $VenvPy -m PyInstaller $Spec --noconfirm --clean --distpath (Join-Path $Root "dist") --workpath (Join-Path $Root "build\pyinstaller")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

# Step 4: copy the example settings and the documentation next to the exe
# (the app creates config.toml / glossary.csv from the examples on first run)
$Dist = Join-Path $Root "dist\LiveSub"
foreach ($f in @("config.example.toml", "glossary.example.csv", "README.md")) {
    Copy-Item $f $Dist -Force
}
Copy-Item (Join-Path $Root "docs") (Join-Path $Dist "docs") -Recurse -Force

# Step 5: zip the whole folder so it can be copied to another PC in one file.
# Antivirus software often locks freshly built files for a few seconds
# ("file is being used by another process"), so retry a few times.
if (-not $NoZip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = Join-Path $Root "dist\LiveSub.zip"
    $done = $false
    for ($try = 1; $try -le 5 -and -not $done; $try++) {
        try {
            if (Test-Path $Zip) { Remove-Item $Zip -Force }
            # includeBaseDirectory = $true: the zip contains a top-level LiveSub folder
            [System.IO.Compression.ZipFile]::CreateFromDirectory($Dist, $Zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
            $done = $true
        } catch {
            Write-Host "Zip attempt $try failed (files may be locked by antivirus), retrying in 5 s ..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }
    if (-not $done) { throw "Could not create $Zip. The exe is ready in $Dist; zip that folder manually." }
    Write-Host "Zip: $Zip" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done: $Dist\LiveSub.exe" -ForegroundColor Green
Write-Host "On the other PC: unzip, double-click LiveSub.exe, paste API keys in the settings window."
