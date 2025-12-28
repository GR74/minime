#!/bin/bash
# Setup script for MiniMe virtual environment
# Usage: bash setup_venv.sh

set -e  # Exit on error

echo "🚀 Setting up MiniMe virtual environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Check if Python 3.11+ is available
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Error: Python 3.11 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements-dev.txt" ]; then
    echo "   Installing with dev dependencies..."
    pip install -r requirements-dev.txt
else
    echo "   Installing core dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "   deactivate"

