"""Configuration management for MiniMe."""

from pathlib import Path
from typing import Optional
import yaml

from minime.schemas import MiniMeConfig


def load_config(config_path: Optional[str] = None) -> MiniMeConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default ./config/identity.yaml

    Returns:
        MiniMeConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if config_path is None:
        config_path = "./config/identity.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        # Return default config if file doesn't exist
        return _get_default_config()

    try:
        return MiniMeConfig.load_from_file(str(config_file))
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}") from e


def _get_default_config() -> MiniMeConfig:
    """Get default configuration for dev mode."""
    return MiniMeConfig(
        vault_path="~/obsidian-vault",
        db_path="./data/minime.db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_cache_size=100,
        default_provider="mock",
        trace_dir="./logs/",
        config_dir="./config/",
        max_context_tokens=4000,
        enable_offline_training=False,
        offline_training_interval_hours=24,
    )


def create_default_config_file(config_path: str) -> None:
    """
    Create a default configuration file.

    Args:
        config_path: Path where to create the config file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    default_config = _get_default_config()

    # Create default identity principles section
    default_yaml = """# MiniMe Configuration

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
    magnitude: 0.9
    decay_rate: 0.05
    scope: global
    tags: [code, style]
  
  - id: reversibility
    name: Reversibility
    description: "Design decisions that can be undone with minimal cost"
    magnitude: 0.8
    decay_rate: 0.05
    scope: global
    tags: [architecture, design]

# System Configuration
vault_path: ~/obsidian-vault
db_path: ./data/minime.db
embedding_model: all-MiniLM-L6-v2
embedding_cache_size: 100
default_provider: mock
trace_dir: ./logs/
config_dir: ./config/
max_context_tokens: 4000
enable_offline_training: false
offline_training_interval_hours: 24

# Risk-based execution settings
safe_paths:
  - ./outputs/
  - ./docs/
  - ./tmp/
system_paths:
  - /usr/
  - /etc/
  - /bin/
allowlisted_commands:
  - git status
  - git log
  - ls
  - pwd
auto_approve_safe: true
low_risk_auto_delay_sec: 2.0
"""

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(default_yaml)

