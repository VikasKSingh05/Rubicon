# Rubicon setup script for Windows PowerShell
Write-Host "Setting up Rubicon environment..." -ForegroundColor Cyan

# 1. Backend dependencies
Write-Host "`n--- Installing Backend Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\backend"
npm.cmd install

# 2. Agent Service dependencies
Write-Host "`n--- Installing Agent Service Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\agent-service"
npm.cmd install

# 3. Frontend dependencies
Write-Host "`n--- Installing Frontend Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\frontend"
npm.cmd install

# 4. AI Engine Python dependencies (optional for local run)
Write-Host "`n--- Installing AI Engine Python Dependencies (optional) ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\ai-engine"
try {
    python -m pip install -r requirements.txt
} catch {
    Write-Host "Note: Heavy native ML dependencies (rasterio, torch) require C++ build tools or Docker. Backend uses deterministic fallback if AI engine is not present locally." -ForegroundColor DarkGray
}

Set-Location "$PSScriptRoot\.."
Write-Host "`nRubicon setup complete!" -ForegroundColor Green
