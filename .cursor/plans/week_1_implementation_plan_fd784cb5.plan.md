---
name: Week 1 Implementation Plan
overview: "Week 1 focuses on establishing the foundational infrastructure: project scaffolding, core data schemas, configuration system, identity layer, basic memory layer (DB + vault indexer), provider interface (mock), CLI skeleton, and testing infrastructure. This creates a working foundation that can run end-to-end with mock providers."
todos:
  - id: setup-project
    content: "Create project scaffolding: pyproject.toml, README.md, Makefile, .gitignore with all dependencies and structure"
    status: completed
  - id: create-schemas
    content: Implement all Pydantic models in minime/schemas.py (Identity, Memory, Mask, Agent, Tool, Trace, Feedback, Config schemas)
    status: completed
    dependencies:
      - setup-project
  - id: create-package-structure
    content: Create all __init__.py files for package structure (identity, memory, graph, mask, context, agents, orchestrator, providers, tools, tracing, feedback)
    status: completed
    dependencies:
      - setup-project
  - id: config-system
    content: Implement minime/config.py with MiniMeConfig class, YAML loading/saving, and default config template
    status: completed
    dependencies:
      - create-schemas
  - id: identity-principles
    content: Implement minime/identity/principles.py with IdentityPrinciple and GlobalIdentityMatrix classes
    status: completed
    dependencies:
      - create-schemas
  - id: identity-loader
    content: Implement minime/identity/loader.py to load identity from YAML and compute embeddings
    status: completed
    dependencies:
      - identity-principles
      - config-system
  - id: db-schema
    content: Implement minime/memory/db.py with SQLite schema (vault_nodes, memory_chunks, graph_edges, graph_proposals) and AsyncDatabase class
    status: pending
    dependencies:
      - create-schemas
  - id: embeddings-module
    content: Implement minime/memory/embeddings.py with EmbeddingModel class using sentence-transformers
    status: completed
    dependencies:
      - setup-project
  - id: chunking-strategy
    content: Implement minime/memory/chunk.py with chunk_note function for note splitting
    status: pending
  - id: vault-indexer
    content: Implement minime/memory/vault.py with VaultIndexer class to parse Obsidian notes, extract metadata, and store in DB
    status: pending
    dependencies:
      - db-schema
      - embeddings-module
      - chunking-strategy
  - id: provider-base
    content: Implement minime/providers/base.py with abstract LLMProvider interface and GenerationResult dataclass
    status: pending
    dependencies:
      - create-schemas
  - id: mock-provider
    content: Implement minime/providers/mock.py with MockProvider class for testing
    status: pending
    dependencies:
      - provider-base
  - id: provider-router
    content: Implement minime/providers/router.py with ProviderRouter class for provider selection
    status: pending
    dependencies:
      - mock-provider
      - config-system
  - id: cli-init
    content: Implement minime/cli.py with init command to set up vault, DB, and config directories
    status: pending
    dependencies:
      - config-system
      - db-schema
  - id: cli-config
    content: Add config_show command to minime/cli.py to display loaded configuration
    status: pending
    dependencies:
      - cli-init
  - id: cli-entry-point
    content: Add CLI entry point to pyproject.toml so minime command is available after install
    status: pending
    dependencies:
      - cli-init
  - id: test-fixtures
    content: Create tests/conftest.py with fixtures for temp_vault, temp_db, mock_config, mock_provider, sample_identity
    status: pending
    dependencies:
      - setup-project
  - id: test-identity
    content: Create tests/test_identity.py with unit tests for identity loading and GlobalIdentityMatrix
    status: pending
    dependencies:
      - identity-loader
      - test-fixtures
  - id: test-db
    content: Create tests/test_db.py with unit tests for database operations
    status: pending
    dependencies:
      - db-schema
      - test-fixtures
  - id: test-vault-indexer
    content: Create tests/test_vault_indexer.py with unit tests for vault indexing
    status: pending
    dependencies:
      - vault-indexer
      - test-fixtures
  - id: test-providers
    content: Create tests/test_providers.py with unit tests for MockProvider
    status: pending
    dependencies:
      - mock-provider
      - test-fixtures
  - id: test-cli
    content: Create tests/test_cli.py with integration tests for CLI commands
    status: pending
    dependencies:
      - cli-init
      - test-fixtures
  - id: tracing-system
    content: Implement minime/tracing/tracer.py and minime/tracing/storage.py for event logging to JSONL
    status: pending
    dependencies:
      - create-schemas
  - id: cli-trace
    content: Add trace_view command to minime/cli.py to display recent trace events
    status: pending
    dependencies:
      - tracing-system
  - id: error-handling
    content: Add comprehensive error handling and validation across all modules
    status: pending
    dependencies:
      - vault-indexer
      - cli-init
      - tracing-system
  - id: documentation
    content: Update README.md with installation and quick start, add docstrings to all public methods
    status: pending
    dependencies:
      - cli-init
  - id: e2e-smoke-test
    content: Create tests/test_e2e_week1.py with end-to-end smoke test of full week 1 flow
    status: pending
    dependencies:
      - cli-init
      - vault-indexer
      - mock-provider
      - tracing-system
---

# Week

1 Implementation Plan: MiniMe Foundation

## Overview

Week 1 establishes the foundational infrastructure needed for the MiniMe system. By the end of the week, we'll have:

- Complete project scaffolding with dependencies
- All core Pydantic schemas defined
- Configuration system with YAML loading
- Identity layer (principles loading/storage)
- SQLite database schema and basic queries
- Vault indexer skeleton (file parsing, metadata extraction)
- Provider interface with mock implementation
- Basic CLI with `init` and `config` commands
- Testing infrastructure with fixtures
- Basic tracing system

## Day 1-2: Project Setup & Core Schemas

### Task 1.1: Project Scaffolding

**Files**: `pyproject.toml`, `README.md`, `Makefile`, `.gitignore`

- Create `pyproject.toml` with dependencies:
- `typer[all]` (CLI framework)
- `pydantic>=2.0` (data validation)
- `pyyaml` (config loading)
- `sentence-transformers` (embeddings)
- `aiosqlite` (async SQLite)
- `httpx` (async HTTP client)
- `pytest` + `pytest-asyncio` (testing)
- Create `README.md` with quick start guide
- Create `Makefile` with commands: `install`, `test`, `lint`, `format`
- Create `.gitignore` (ignore `data/`, `logs/`, `checkpoints/`, `.env`, `venv/`, `__pycache__/`)

### Task 1.2: Core Data Schemas

**File**: `minime/schemas.py`Implement all Pydantic models from section 5 of the plan:

- `IdentityPrinciple`, `GlobalIdentityMatrix`
- `VaultNode`, `GraphEdge`, `GraphUpdateProposal`, `MemoryChunk`
- `RetrievalQuery`, `RetrievalResult`
- `MaskWeights`, `MaskNetworkInput`, `MaskNetworkOutput`
- `AgentMessage`, `AgentSpec`, `ToolDefinition`, `ToolCall`, `ToolResult`
- `ActionProposal`
- `TraceEvent`, `RunTrace`
- `FeedbackCorrectionPair`
- `MiniMeConfig`

**Key requirements**:

- All models use Pydantic v2 syntax
- Proper field validation and defaults
- JSON serialization support
- Type hints throughout

### Task 1.3: Package Structure

**Files**: Create all `__init__.py` files for package structureCreate empty `__init__.py` files in:

- `minime/`
- `minime/identity/`
- `minime/memory/`
- `minime/graph/`
- `minime/mask/`
- `minime/context/`
- `minime/agents/`
- `minime/orchestrator/`
- `minime/providers/`
- `minime/tools/`
- `minime/tracing/`
- `minime/feedback/`
- `tests/`

## Day 2-3: Configuration & Identity Layer

### Task 2.1: Configuration System

**File**: `minime/config.py`

- Implement `MiniMeConfig` class with:
- `load_from_file(path: str) -> MiniMeConfig` (loads from YAML)
- `save_to_file(path: str)` (saves to YAML)
- Validation of all config fields
- Default values for dev mode
- Create default config template in `config/identity.yaml` (from section 9a of plan)

### Task 2.2: Identity Layer - Principles

**File**: `minime/identity/principles.py`

- Implement `IdentityPrinciple` class (from schemas)
- Implement `GlobalIdentityMatrix` class with:
- `to_dict() -> Dict[str, Any]` (returns embedding dict)
- `get_principle(id: str) -> Optional[IdentityPrinciple]`
- `add_principle(principle: IdentityPrinciple)`
- `update_principle(id: str, updates: Dict)`

### Task 2.3: Identity Layer - Loader

**File**: `minime/identity/loader.py`

- Implement `load_identity_from_yaml(path: str) -> GlobalIdentityMatrix`
- Parses YAML config
- Creates `IdentityPrinciple` objects
- Computes embeddings for principles (using sentence-transformers)
- Returns `GlobalIdentityMatrix`
- Handle missing/empty config gracefully

## Day 3-4: Memory Layer Foundation

### Task 3.1: SQLite Database Schema

**File**: `minime/memory/db.py`

- Implement `init_db(db_path: str)` function that creates:
- `vault_nodes` table (node_id, path, title, frontmatter JSON, tags JSON, domain, scope, links JSON, backlinks JSON, created_at, updated_at, embedding_ref)
- `memory_chunks` table (chunk_id, node_id, content, embedding JSON, metadata JSON, position)
- `graph_edges` table (edge_id, source_node_id, target_node_id, edge_type, weight, rationale, confidence, is_approved, created_at, approved_at)
- `graph_proposals` table (proposal_id, edges_to_add JSON, edges_to_remove JSON, confidence, requires_user_approval, rationale, created_at)
- Implement `AsyncDatabase` class with basic methods:
- `async def insert_node(node: VaultNode) -> str`
- `async def get_node(node_id: str) -> Optional[VaultNode]`
- `async def count_nodes() -> int`
- `async def get_pending_proposals() -> List[GraphUpdateProposal]`

### Task 3.2: Embeddings Module

**File**: `minime/memory/embeddings.py`

- Implement `EmbeddingModel` class:
- `__init__(model_name: str = "all-MiniLM-L6-v2")`
- `encode(texts: List[str]) -> List[List[float]]` (batch encoding)
- `encode_single(text: str) -> List[float]`
- Support both sentence-transformers (local) and OpenAI embeddings (stub for now)

### Task 3.3: Chunking Strategy

**File**: `minime/memory/chunk.py`

- Implement `chunk_note(note_text: str, max_tokens: int = 512, overlap: int = 128) -> List[str]`
- Preserve frontmatter as metadata (not in chunk content)
- Handle edge cases (empty notes, very short notes)

### Task 3.4: Vault Indexer Skeleton

**File**: `minime/memory/vault.py`

- Implement `VaultIndexer` class:
- `__init__(vault_path: str, db: AsyncDatabase, embedding_model: EmbeddingModel)`
- `async def index() -> List[VaultNode]`:
    - Scans vault directory for `.md` files
    - Parses frontmatter (YAML)
    - Extracts wikilinks `[[...]]`
    - Extracts tags (from frontmatter and inline `#tag`)
    - Creates `VaultNode` objects
    - Computes embeddings for each note
    - Stores in database
    - Returns list of nodes
- Handle empty vault gracefully (returns empty list)
- Handle file read errors gracefully

## Day 4-5: Provider Interface & CLI

### Task 4.1: Provider Base Interface

**File**: `minime/providers/base.py`

- Implement abstract `LLMProvider` class:
- `async def generate(prompt: str, system: str, tools: List[ToolDefinition], model: str, max_tokens: int, temperature: float, structured_output_schema: Optional[Dict] = None) -> GenerationResult`
- Define `GenerationResult` dataclass:
- `text: str`
- `tool_calls: List[ToolCall]`
- `structured: Optional[Dict]`
- `stop_reason: str`

### Task 4.2: Mock Provider

**File**: `minime/providers/mock.py`

- Implement `MockProvider(LLMProvider)`:
- Returns hardcoded responses for testing
- Supports structured output (returns mock JSON)
- Supports tool calling (returns mock tool calls)
- Logs all calls for debugging

### Task 4.3: Provider Router Skeleton

**File**: `minime/providers/router.py`

- Implement `ProviderRouter` class:
- `__init__(config: MiniMeConfig)`
- `select_provider(agent_name: str, task_domain: Optional[str], task_complexity: str) -> LLMProvider`
- MVP: simple routing (returns mock for all agents)

### Task 4.4: CLI Entry Point - Init Command

**File**: `minime/cli.py`

- Set up Typer app
- Implement `init` command:
- Creates vault directory (if doesn't exist)
- Creates `data/` directory
- Initializes SQLite database (calls `init_db`)
- Creates `config/` directory
- Creates default `config/identity.yaml` (from section 9a)
- Creates `logs/` directory
- Prints success message

### Task 4.5: CLI Entry Point - Config Command

**File**: `minime/cli.py`

- Implement `config_show` command:
- Loads config from `config/identity.yaml`
- Pretty-prints config as JSON
- Handles missing config gracefully

### Task 4.6: CLI Entry Point Setup

**File**: `pyproject.toml` (entry point)

- Add entry point: `minime = "minime.cli:app"`

## Day 5: Testing Infrastructure

### Task 5.1: Test Fixtures

**File**: `tests/conftest.py`

- Create fixtures:
- `temp_vault` (temporary Obsidian vault with sample notes)
- `temp_db` (temporary SQLite database)
- `mock_config` (MiniMeConfig for testing)
- `mock_provider` (MockProvider instance)
- `sample_identity` (GlobalIdentityMatrix with test principles)

### Task 5.2: Unit Tests - Identity Layer

**File**: `tests/test_identity.py`

- Test `load_identity_from_yaml` (loads from YAML, computes embeddings)
- Test `GlobalIdentityMatrix.to_dict()` (returns correct structure)
- Test `GlobalIdentityMatrix.get_principle()` (retrieves by ID)

### Task 5.3: Unit Tests - Database

**File**: `tests/test_db.py`

- Test `init_db` (creates all tables)
- Test `insert_node` (stores node, retrieves correctly)
- Test `count_nodes` (returns correct count, handles empty DB)

### Task 5.4: Unit Tests - Vault Indexer

**File**: `tests/test_vault_indexer.py`

- Test `VaultIndexer.index()` with sample vault:
- Parses frontmatter correctly
- Extracts wikilinks
- Extracts tags
- Creates embeddings
- Stores in DB
- Test empty vault handling (returns empty list, doesn't crash)

### Task 5.5: Unit Tests - Mock Provider

**File**: `tests/test_providers.py`

- Test `MockProvider.generate()` (returns GenerationResult)
- Test structured output (returns valid JSON)
- Test tool calling (returns tool calls)

### Task 5.6: Integration Test - CLI Init

**File**: `tests/test_cli.py`

- Test `minime init` command:
- Creates all directories
- Creates database
- Creates config file
- Can be run multiple times (idempotent)

## Day 6-7: Tracing & Polish

### Task 6.1: Tracing System

**File**: `minime/tracing/tracer.py`

- Implement `Tracer` class:
- `start_run(task_query: str) -> str` (returns run_id)
- `log_event(run_id: str, event_type: str, payload: Dict, agent_name: Optional[str] = None)`
- `end_run(run_id: str, final_output: str, success: bool) -> RunTrace`
- Implement `TraceStorage` class:
- `append_event(event: TraceEvent)` (writes to JSONL)
- `get_recent_events(n: int = 10) -> List[TraceEvent]`
- `get_run_trace(run_id: str) -> Optional[RunTrace]`

### Task 6.2: CLI Trace Command

**File**: `minime/cli.py`

- Implement `trace_view` command:
- Shows last N events from `logs/traces.jsonl`
- Pretty-prints JSON
- Handles empty/missing trace file

### Task 6.3: Error Handling & Validation

**Files**: All modules

- Add proper error handling:
- File not found errors
- Database errors
- Invalid YAML/config errors
- Missing dependencies
- Add input validation for all public methods
- Add helpful error messages

### Task 6.4: Documentation

**Files**: `README.md`, docstrings

- Update `README.md` with:
- Installation instructions
- Quick start guide
- Week 1 features
- Add docstrings to all public classes/methods
- Add type hints throughout

### Task 6.5: End-to-End Smoke Test

**File**: `tests/test_e2e_week1.py`

- Test full flow:

1. `minime init` creates everything
2. Load identity from config
3. Index empty vault (should not crash)
4. Create mock provider
5. Generate a simple response
6. Log trace event
7. View trace

## Deliverables Checklist

By end of Week 1, verify:

- [ ] `minime init` command works and creates all directories/files
- [ ] `minime config` command shows loaded config
- [ ] Identity principles can be loaded from YAML
- [ ] SQLite database schema is created correctly
- [ ] Vault indexer can parse sample Obsidian notes
- [ ] Mock provider returns structured responses
- [ ] Tracing system logs events to JSONL
- [ ] All unit tests pass
- [ ] CLI can be installed via `pip install -e .`
- [ ] Project structure matches repo scaffold from section 8

## Success Criteria

Week 1 is complete when:

1. A developer can run `minime init` and get a working setup
2. All core schemas are defined and importable
3. Identity layer can load/store principles
4. Memory layer can index a vault (even if empty)
5. Mock provider can be used for testing
6. Basic tracing works
7. Test suite runs and passes

## Notes

- Focus on getting the foundation right - don't rush to full features
- All async code should use proper `asyncio` patterns