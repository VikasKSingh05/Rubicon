# Rubicon setup script for Windows PowerShell
Write-Host "Setting up Rubicon environment..." -ForegroundColor Cyan

# 1. Backend dependencies
Write-Host "`n--- Installing Backend Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\backend"
npm install

# 2. Agent Service dependencies
Write-Host "`n--- Installing Agent Service Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\agent-service"
npm install

# 3. Frontend dependencies
Write-Host "`n--- Installing Frontend Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\frontend"
npm install

# 4. AI Engine Python dependencies
Write-Host "`n--- Installing AI Engine Python Dependencies ---" -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\ai-engine"
python -m pip install -r requirements.txt

Set-Location "$PSScriptRoot\.."
Write-Host "`nRubicon setup complete!" -ForegroundColor Green
