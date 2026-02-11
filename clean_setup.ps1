# Clean setup script for AARES project
Write-Host "Setting up AARES Project Environment..." -ForegroundColor Cyan

# Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists."
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
try {
    . .venv\Scripts\Activate.ps1
} catch {
    Write-Error "Failed to activate virtual environment. Is execution policy set? (Try: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)"
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Error "Failed to install dependencies."
        exit 1
    }
} else {
    Write-Warning "requirements.txt not found. Skipping dependency installation."
}

Write-Host "Setup complete! To activate the environment manually, run: .venv\Scripts\Activate.ps1" -ForegroundColor Green
