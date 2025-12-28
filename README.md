# MiniMe: Identity-Conditioned LLM Orchestration System

MiniMe is an identity-conditioned LLM orchestration system that combines personal identity principles, memory from Obsidian vaults, and multi-agent orchestration to create a personalized AI assistant.

## Features (Week 1 - Day 1-2)

- **Project Scaffolding**: Complete project structure with dependencies
- **Core Data Schemas**: All Pydantic models defined
- **Configuration System**: YAML-based configuration loading
- **Identity Layer**: Principles storage and loading with embeddings

## Quick Start

### Installation

#### Option 1: Using requirements.txt (Recommended for beginners)

```bash
# Clone the repository
git clone <repo-url>
cd minime

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For development (includes testing/linting tools):
pip install -r requirements-dev.txt
```

#### Option 2: Using pyproject.toml (Modern Python packaging)

```bash
# Clone the repository
git clone <repo-url>
cd minime

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

#### Option 3: Using Makefile

```bash
# Create venv and install everything
make install
```

### Initialize MiniMe

```bash
# Initialize the system (creates vault, DB, config)
minime init

# View configuration
minime config
```

## Project Structure

```
minime/
├── minime/           # Main package
│   ├── identity/     # Identity principles
│   ├── memory/       # Vault indexing & storage
│   ├── providers/    # LLM provider interfaces
│   └── cli.py        # CLI entry point
├── config/           # Configuration files
└── tests/            # Test suite
```

## License

MIT

