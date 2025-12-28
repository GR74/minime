# Explanation: `minime/config.py`

This file handles loading and managing MiniMe's configuration from YAML files.

## Overview

The `config.py` module provides functions to:
- Load configuration from YAML files
- Create default configuration files
- Provide sensible defaults when config is missing

It works with the `MiniMeConfig` schema defined in `schemas.py`.

---

## File Structure

```python
# Functions
- load_config(config_path)           # Load config from file
- _get_default_config()              # Get default config (private)
- create_default_config_file(path)   # Create default config file
```

---

## Detailed Explanation

### Function: `load_config(config_path: Optional[str] = None) -> MiniMeConfig`

**Purpose**: Load configuration from a YAML file, with fallback to defaults.

**Parameters:**
- `config_path`: Optional path to config file
  - If `None`: Uses default `"./config/identity.yaml"`
  - Can specify custom path: `"/path/to/config.yaml"`

**Returns:**
- `MiniMeConfig` object with loaded settings

**How it works:**

```python
def load_config(config_path: Optional[str] = None) -> MiniMeConfig:
    # 1. Use default path if not provided
    if config_path is None:
        config_path = "./config/identity.yaml"
    
    # 2. Check if file exists
    config_file = Path(config_path)
    if not config_file.exists():
        # 3. Return defaults if file doesn't exist
        return _get_default_config()
    
    # 4. Try to load from file
    try:
        return MiniMeConfig.load_from_file(str(config_file))
    except Exception as e:
        # 5. Raise error if loading fails
        raise ValueError(f"Failed to load config: {e}") from e
```

**Key Features:**

1. **Graceful Degradation**: If file doesn't exist, returns defaults (doesn't crash)
2. **Error Handling**: Catches exceptions and provides helpful error messages
3. **Flexible Paths**: Can use default or custom path

**Example Usage:**

```python
from minime.config import load_config

# Use default path
config = load_config()  # Loads from ./config/identity.yaml

# Use custom path
config = load_config("/path/to/my-config.yaml")

# If file doesn't exist, gets defaults
config = load_config("./missing-file.yaml")  # Returns default config
```

---

### Function: `_get_default_config() -> MiniMeConfig`

**Purpose**: Create a default configuration for development mode.

**Note**: The `_` prefix means it's a "private" function (internal use only).

**Returns:**
- `MiniMeConfig` with sensible defaults

**Default Values:**

```python
MiniMeConfig(
    vault_path="~/obsidian-vault",        # Your Obsidian vault
    db_path="./data/minime.db",          # SQLite database
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    embedding_cache_size=100,            # Cache size in MB
    default_provider="mock",            # LLM provider (mock for dev)
    trace_dir="./logs/",                 # Where to store logs
    config_dir="./config/",              # Where config files are
    max_context_tokens=4000,            # Token limit for LLM
    enable_offline_training=False,      # Training disabled by default
    offline_training_interval_hours=24,  # How often to train
)
```

**Why Defaults?**

1. **Development**: System works out-of-the-box
2. **Testing**: Tests can run without config files
3. **Fallback**: If config is missing, system still works

**Example:**

```python
# System starts, no config file exists
config = load_config()  # Returns default config
# System works with defaults!
```

---

### Function: `create_default_config_file(config_path: str) -> None`

**Purpose**: Create a new configuration file with default values and example identity principles.

**Parameters:**
- `config_path`: Where to create the config file
  - Example: `"./config/identity.yaml"`

**What it does:**

1. **Creates directory** if it doesn't exist
2. **Writes default YAML** with:
   - Example identity principles (Modularity, Clarity, Reversibility)
   - System configuration
   - Risk-based execution settings

**The Default YAML Structure:**

```yaml
# Identity Principles
principles:
  - id: modularity
    name: Modularity
    description: "Break systems into independent, testable modules"
    magnitude: 1.0
    decay_rate: 0.05
    scope: global
    tags: [architecture, code]
  
  - id: clarity
    name: Clarity Over Cleverness
    description: "Prefer clear, obvious code to clever optimizations"
    # ... more fields

# System Configuration
vault_path: ~/obsidian-vault
db_path: ./data/minime.db
embedding_model: all-MiniLM-L6-v2
# ... more settings

# Risk-based execution settings
safe_paths:
  - ./outputs/
  - ./docs/
  - ./tmp/
system_paths:
  - /usr/
  - /etc/
  - /bin/
```

**Example Usage:**

```python
from minime.config import create_default_config_file

# Create default config file
create_default_config_file("./config/identity.yaml")
# File created with defaults and example principles
```

**When is this used?**

- During `minime init` command
- When setting up a new MiniMe installation
- When user wants to reset their config

---

## How It Works Together

### Flow: Loading Configuration

```
1. User runs: minime init
   ↓
2. create_default_config_file() creates ./config/identity.yaml
   ↓
3. User runs: minime task "something"
   ↓
4. load_config() called
   ↓
5. Checks if ./config/identity.yaml exists
   ↓
6a. If exists: Loads from file
6b. If not: Returns defaults
   ↓
7. MiniMeConfig object ready to use
```

### Flow: First Time Setup

```
1. User installs MiniMe
   ↓
2. Runs: minime init
   ↓
3. create_default_config_file() creates:
   - ./config/identity.yaml (with example principles)
   - ./data/ directory
   - ./logs/ directory
   ↓
4. User can now:
   - Edit ./config/identity.yaml to customize
   - Add their own principles
   - Change system settings
```

---

## Configuration Fields Explained

### Identity Principles Section

```yaml
principles:
  - id: modularity          # Unique identifier
    name: Modularity        # Display name
    description: "..."      # What it means
    magnitude: 1.0          # Importance (0.0-1.0)
    decay_rate: 0.05        # How fast it adapts
    scope: global           # Where it applies
    tags: [architecture]    # Categories
```

### System Configuration

```yaml
vault_path: ~/obsidian-vault      # Your Obsidian vault location
db_path: ./data/minime.db         # SQLite database file
embedding_model: all-MiniLM-L6-v2 # Which embedding model
default_provider: mock            # LLM provider (mock/openai/anthropic)
max_context_tokens: 4000          # Token limit per request
```

### Risk-Based Execution

```yaml
safe_paths:          # Paths where writes are low-risk
  - ./outputs/
  - ./docs/

system_paths:        # Paths that are always high-risk
  - /usr/
  - /etc/

allowlisted_commands: # Commands that are medium-risk
  - git status
  - ls
```

**Why risk-based?**
- MiniMe can write files and run commands
- We need to prevent dangerous operations
- Safe paths = OK to write automatically
- System paths = Always require approval

---

## Integration with Other Modules

### 1. Identity Loader

```python
# In minime/identity/loader.py
from minime.config import load_config

config = load_config()
# Uses config.embedding_model to create EmbeddingModel
```

### 2. CLI Commands

```python
# In minime/cli.py
from minime.config import load_config, create_default_config_file

@app.command()
def init():
    create_default_config_file("./config/identity.yaml")
    # Creates default config

@app.command()
def config_show():
    config = load_config()
    print(config.model_dump_json(indent=2))
    # Shows current config
```

### 3. Database Setup

```python
# In minime/memory/db.py
from minime.config import load_config

config = load_config()
# Uses config.db_path to initialize database
```

---

## Error Handling

### Missing Config File

```python
config = load_config("./missing.yaml")
# Returns default config (doesn't crash)
```

### Invalid YAML

```python
# If YAML is malformed
config = load_config("./bad-config.yaml")
# Raises ValueError with helpful message
```

### Missing Fields

```python
# If YAML is missing fields
config = load_config("./partial-config.yaml")
# Uses defaults for missing fields (Pydantic handles this)
```

---

## Best Practices

### 1. Always Provide Defaults

```python
# Good: Has defaults
config = load_config()  # Works even if file missing

# Bad: Assumes file exists
config = MiniMeConfig.load_from_file("./config.yaml")  # Crashes if missing
```

### 2. Use Path Expansion

```python
# Config uses ~/obsidian-vault
# Path.expanduser() converts ~ to home directory
vault_path = Path(config.vault_path).expanduser()
```

### 3. Validate After Loading

```python
config = load_config()
if not Path(config.vault_path).exists():
    print("Warning: Vault path doesn't exist")
```

---

## Summary

The `config.py` module is the **configuration manager** for MiniMe:

1. **Loads config** from YAML files
2. **Provides defaults** when config is missing
3. **Creates default files** for new installations
4. **Handles errors** gracefully

**Key Benefits:**
- ✅ System works out-of-the-box (defaults)
- ✅ Easy to customize (edit YAML)
- ✅ No crashes if config missing
- ✅ Clear error messages if invalid

Without this module, you'd have to:
- Hardcode all settings
- Manually create config files
- Handle errors yourself
- No way to customize

With this module, you get:
- ✅ Flexible configuration
- ✅ Sensible defaults
- ✅ Easy setup
- ✅ Error handling

