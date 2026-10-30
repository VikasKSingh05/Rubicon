# Rubicon multi-service test runner for PowerShell
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Write-Host "Running Rubicon full test suite..." -ForegroundColor Cyan

$failed = 0

Write-Host "`n1/4: AI Engine (pytest)..." -ForegroundColor Yellow
Push-Location "$Root\ai-engine"
try {
    python -m pytest
    Write-Host "AI Engine tests passed!" -ForegroundColor Green
} catch {
    Write-Host "AI Engine tests failed!" -ForegroundColor Red
    $failed = 1
} finally {
    Pop-Location
}

Write-Host "`n2/4: Backend tests (node --test)..." -ForegroundColor Yellow
Push-Location "$Root\backend"
try {
    npm test
    Write-Host "Backend tests passed!" -ForegroundColor Green
} catch {
    Write-Host "Backend tests failed!" -ForegroundColor Red
    $failed = 1
} finally {
    Pop-Location
}

Write-Host "`n3/4: Agent Service tests (node --test)..." -ForegroundColor Yellow
Push-Location "$Root\agent-service"
try {
    npm test
    Write-Host "Agent Service tests passed!" -ForegroundColor Green
} catch {
    Write-Host "Agent Service tests failed!" -ForegroundColor Red
    $failed = 1
} finally {
    Pop-Location
}

Write-Host "`n4/4: Frontend tests (vitest)..." -ForegroundColor Yellow
Push-Location "$Root\frontend"
try {
    npm test
    Write-Host "Frontend tests passed!" -ForegroundColor Green
} catch {
    Write-Host "Frontend tests failed!" -ForegroundColor Red
    $failed = 1
} finally {
    Pop-Location
}

if ($failed -ne 0) {
    Write-Host "`nSome tests failed." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nAll test suites passed cleanly!" -ForegroundColor Green
}
