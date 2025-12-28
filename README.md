# MiniMe: Identity-Conditioned LLM Orchestration System

MiniMe is an identity-conditioned LLM orchestration system that combines personal identity principles, memory from Obsidian vaults, and multi-agent orchestration to create a personalized AI assistant.

## Features (Week 1 - Day 1-2)

- **Project Scaffolding**: Complete project structure with dependencies
- **Core Data Schemas**: All Pydantic models defined
- **Configuration System**: YAML-based configuration loading
- **Identity Layer**: Principles storage and loading with embeddings

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd minime

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
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

