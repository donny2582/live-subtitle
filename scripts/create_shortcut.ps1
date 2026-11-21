<#
.SYNOPSIS
    Create a desktop shortcut that starts LiveSub with no console window.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
.NOTES
    The shortcut runs ".venv\Scripts\pythonw.exe -m livesub". pythonw is the
    windowless Python, so no black console window appears; errors go to
    logs\livesub.log (and fatal ones are shown in a message box).
    Messages are ASCII on purpose: Windows PowerShell 5.1 reads BOM-less
    scripts using the system code page (Big5 on zh-TW), which garbles CJK text.
    setup.ps1 calls this script automatically at the end.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$VenvPyw = Join-Path $Root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $VenvPyw)) { throw "Run scripts\setup.ps1 first." }

# 1) Draw the app icon into an .ico file (same blue icon as the tray)
$Icon = Join-Path $Root ".venv\livesub.ico"
& $VenvPy -c "import sys; from livesub.ui.icon import write_ico; write_ico(sys.argv[1])" $Icon
if ($LASTEXITCODE -ne 0) { throw "Failed to create the icon file" }

# 2) Shortcut name built from code points (U+5373 U+6642 U+5B57 U+5E55) so this file stays ASCII
$Name = -join [char[]](0x5373, 0x6642, 0x5B57, 0x5E55)
# GetFolderPath also handles a Desktop redirected to OneDrive
$Desktop = [Environment]::GetFolderPath("Desktop")
$LinkPath = Join-Path $Desktop "$Name.lnk"

# 3) Create (or overwrite) the shortcut via the Windows Script Host COM object
$Shell = New-Object -ComObject WScript.Shell
$Link = $Shell.CreateShortcut($LinkPath)
$Link.TargetPath = $VenvPyw
$Link.Arguments = "-m livesub"
$Link.WorkingDirectory = $Root
$Link.IconLocation = "$Icon,0"
$Link.Description = "LiveSub live subtitles"
$Link.Save()

Write-Host "Desktop shortcut created: $LinkPath" -ForegroundColor Green
