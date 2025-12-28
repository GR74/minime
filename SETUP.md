# MiniMe Setup Guide

This guide will help you set up MiniMe on your system.

## Prerequisites

- **Python 3.11 or higher** (3.12 recommended)
- **pip** (usually comes with Python)
- **Git** (to clone the repository)

### Check Your Python Version

```bash
python3 --version
# or
python --version
```

You should see `Python 3.11.x` or higher.

---

## Setup Methods

### Method 1: Automated Setup Script (Easiest)

#### On macOS/Linux:

```bash
bash setup_venv.sh
```

This script will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set everything up automatically

#### On Windows (PowerShell):

```powershell
.\setup_venv.ps1
```

---

### Method 2: Manual Setup (Step-by-Step)

#### Step 1: Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv

# On Windows
python -m venv venv
```

#### Step 2: Activate Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows (Command Prompt)
venv\Scripts\activate

# On Windows (PowerShell)
venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error on Windows PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

#### Step 4: Install Dependencies

**Option A: Using requirements.txt (Recommended)**

```bash
# Core dependencies only
pip install -r requirements.txt

# With development tools (testing, linting, etc.)
pip install -r requirements-dev.txt
```

**Option B: Using pyproject.toml**

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

### Method 3: Using Makefile

```bash
# Create venv (first time only)
make setup

# Activate venv manually, then:
make install
```

---

## Verify Installation

After installation, verify everything works:

```bash
# Check if minime command is available
minime --help

# Or if not installed as command:
python -m minime.cli --help
```

---

## Troubleshooting

### Python Version Issues

**Problem**: `python3` command not found

**Solution**: 
- On Windows, use `python` instead of `python3`
- Make sure Python 3.11+ is installed
- Add Python to your PATH

### Virtual Environment Issues

**Problem**: `venv` module not found

**Solution**:
```bash
# Install venv module
python3 -m pip install --user virtualenv
```

### Permission Errors (macOS/Linux)

**Problem**: Permission denied when activating venv

**Solution**:
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### PowerShell Execution Policy (Windows)

**Problem**: Cannot activate venv in PowerShell

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors After Installation

**Problem**: Module not found errors

**Solution**:
1. Make sure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements-dev.txt`
3. Install in editable mode: `pip install -e .`

---

## Development Setup

For development, you'll want the dev dependencies:

```bash
# Activate venv first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Or use pyproject.toml
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `pytest-asyncio` - Async test support
- `black` - Code formatter
- `ruff` - Fast linter
- `mypy` - Type checker

---

## Next Steps

After setup:

1. **Initialize MiniMe**:
   ```bash
   minime init
   ```

2. **View configuration**:
   ```bash
   minime config
   ```

3. **Run tests** (if dev dependencies installed):
   ```bash
   make test
   # or
   pytest
   ```

---

## Virtual Environment Management

### Activating

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Deactivating

```bash
deactivate
```

### Removing

```bash
# Just delete the venv folder
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

---

## Alternative: Using pyenv

If you use `pyenv` for Python version management:

```bash
# Install Python 3.11
pyenv install 3.11.9

# Set local version
pyenv local 3.11.9

# Create venv
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

The `.python-version` file in the repo will tell pyenv which version to use.

---

## Need Help?

- Check the [README.md](README.md) for quick start
- Review [docs/](docs/) for detailed explanations
- Open an issue on GitHub

Happy coding! 🚀

