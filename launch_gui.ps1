Write-Host "Starting Antigravity Language Model Desktop GUI..." -ForegroundColor Cyan
Set-Location -Path $PSScriptRoot
& ".\.venv\Scripts\python.exe" "scripts\gui.py"
