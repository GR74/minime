# Setup script for MiniMe virtual environment (Windows PowerShell)
# Usage: .\setup_venv.ps1

Write-Host "🚀 Setting up MiniMe virtual environment..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "📌 $pythonVersion" -ForegroundColor Yellow

# Check if Python 3.11+ is available
try {
    $versionOutput = python -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error: Could not run Python" -ForegroundColor Red
        exit 1
    }
    $version = $versionOutput.ToString().Trim()
    $major, $minor = $version -split '\.'
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
        Write-Host "❌ Error: Python 3.11 or higher is required" -ForegroundColor Red
        Write-Host "   Current version: $version" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Python version check passed: $version" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Could not determine Python version" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📥 Installing dependencies..." -ForegroundColor Cyan
if (Test-Path "requirements-dev.txt") {
    Write-Host "   Installing with dev dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
} else {
    Write-Host "   Installing core dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Cyan
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate, run:" -ForegroundColor Cyan
Write-Host "   deactivate" -ForegroundColor White

