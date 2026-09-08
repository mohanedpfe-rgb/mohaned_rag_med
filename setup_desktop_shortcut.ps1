$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "BookRAG Studio.lnk"
$launcherPath = Join-Path $projectRoot "launch_bookrag.bat"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = "Open BookRAG Studio"
# Use a clear built-in application icon so the shortcut is recognizable immediately.
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,167"
$shortcut.Save()

Write-Host "Created desktop shortcut: $shortcutPath"
