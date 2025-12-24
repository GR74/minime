# MiniMe: Technical Architecture & Implementation Plan

**Project**: MiniMe / Mini Me  
**Type**: Identity-Conditioned LLM Orchestration System  
**Status**: Build-Ready Implementation Plan  
**Date**: December 24, 2025

---

## Table of Contents

1. [One-Page System Map](#1-one-page-system-map)
2. [MVP Scope (2–3 weeks)](#2-mvp-scope-23-weeks)
3. [Tech Stack + Why](#3-tech-stack--why)
4. [API + Tooling Plan](#4-api--tooling-plan)
5. [Core Data Schemas](#5-core-data-schemas-pydantic-models)
6. [Mask System Implementation Plan](#6-mask-system-implementation-plan)
7. [Multi-Agent Orchestration Plan](#7-multi-agent-orchestration-plan)
8. [Repo Scaffold](#8-repo-scaffold-exact-tree)
9. [Starter Code](#9-starter-code-runnable)
10. [Testing Strategy](#10-testing-strategy)
11. [Next 10 Upgrades](#11-next-10-upgrades-ranked-by-roi)
12. [Summary & Next Steps](#summary--next-steps)

---

## 1) One-Page System Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MINIME SYSTEM MAP                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USER INPUT (CLI / VS Code)                                         │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         TASK CLASSIFIER & ROUTING LAYER                     │  │
│  │  - Infer domain + complexity                                 │  │
│  │  - Select agent team                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                            │
│         ├─────────────────┬─────────────────┬────────────────────┤  │
│         ▼                 ▼                 ▼                    ▼  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐
│  │IDENTITY LAYER│ │MEMORY LAYER  │ │MASK SYSTEM   │ │ORCHESTRATOR │
│  ├──────────────┤ ├──────────────┤ ├──────────────┤ ├─────────────┤
│  │ P_global     │ │ Obsidian     │ │ Mask Network │ │ Context     │
│  │ (vectors)    │ │ Vault        │ │ (MLP)        │ │ Broker      │
│  │ Magnitude    │ │              │ │ Weighting    │ │ Agent Loop  │
│  │ Decay        │ │ ─────────────│ │              │ │ Integration │
│  │ Scope        │ │ VaultIndexer │ │ ─────────────│ │             │
│  │              │ │ (watches FS) │ │ Hierarchical │ │ ─────────────
│  │              │ │ Embeddings   │ │ (Global/     │ │ Provider    │
│  │              │ │              │ │ Domain/Task/ │ │ Router      │
│  │              │ │ ─────────────│ │ Agent)       │ │             │
│  │              │ │ Graph Store  │ │              │ │ ─────────────
│  │              │ │ (SQLite)     │ │              │ │ Agents:     │
│  │              │ │              │ │              │ │ - Architect │
│  │              │ │ ─────────────│ │              │ │ - Research  │
│  │              │ │ ContextMgr   │ │              │ │ - Builder   │
│  │              │ │ (retrieval + │ │              │ │ - Critic    │
│  │              │ │  graph-aware │ │              │ │ - Integrator
│  │              │ │  + gating)   │ │              │ │             │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘
│         │                 │                 │             │         │
│         └─────────────────┴─────────────────┴─────────────┘         │
│                           │                                         │
│                           ▼                                         │
│         ┌───────────────────────────────────────────────┐           │
│         │  CONTEXT ASSEMBLY (Context Broker)            │           │
│         │  - Identity Block (weighted by mask)          │           │
│         │  - Domain Constraints (weighted)              │           │
│         │  - Retrieved Memory (graph-aware, weighted)   │           │
│         │  - Task Instructions                          │           │
│         │  - Safety/Output Constraints                  │           │
│         │  - Token Budget Enforcement                   │           │
│         └───────────────────────────────────────────────┘           │
│                           │                                         │
│                           ▼                                         │
│         ┌───────────────────────────────────────────────┐           │
│         │  LLM GENERATION (Frozen Model)                │           │
│         │  - OpenAI-compatible / Anthropic / Local      │           │
│         │  - Structured output (JSON schemas)           │           │
│         │  - Tool calling                               │           │
│         └───────────────────────────────────────────────┘           │
│                           │                                         │
│                           ▼                                         │
│         ┌───────────────────────────────────────────────┐           │
│         │  OUTPUT PROCESSING & ACTION PROPOSAL           │           │
│         │  - Parse structured LLM output                │           │
│         │  - Propose actions (write/run/patch)          │           │
│         │  - Wait for user approval (GATED)             │           │
│         └───────────────────────────────────────────────┘           │
│                           │                                         │
│                    ┌──────┴──────┐                                  │
│                    ▼             ▼                                  │
│              ┌─────────┐    ┌──────────────┐                        │
│              │ EXECUTE │    │ COLLECT      │                        │
│              │ TOOLS   │    │ FEEDBACK     │                        │
│              │ (Gated) │    │ (Explicit/   │                        │
│              │         │    │  Implicit)   │                        │
│              └────┬────┘    └──────┬───────┘                        │
│                   │                │                                │
│                   └────────┬───────┘                                │
│                            ▼                                        │
│         ┌───────────────────────────────────────────────┐           │
│         │  TRACE LOGGING (JSONL)                        │           │
│         │  - All retrievals, masks, agent outputs       │           │
│         │  - Token budgets, decisions                   │           │
│         │  - Feedback signals                           │           │
│         └───────────────────────────────────────────────┘           │
│                            │                                        │
│                            ▼                                        │
│         ┌───────────────────────────────────────────────┐           │
│         │  OFFLINE TRAINING (Background Job)             │           │
│         │  - Mask network fine-tuning                   │           │
│         │  - Embedding refresh                          │           │
│         │  - Graph proposal generation                  │           │
│         │  (No updates to frozen models)                │           │
│         └───────────────────────────────────────────────┘           │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ ONLINE vs OFFLINE:                                                   │
│  - ONLINE: Task classification, mask generation, context assembly,  │
│    LLM generation, action approval, tool execution, tracing         │
│  - OFFLINE: Vault reindexing, embedding refresh, mask training,     │
│    graph proposal generation, feedback integration                  │
├─────────────────────────────────────────────────────────────────────┤
│ PERSISTENT STORAGE:                                                  │
│  - Obsidian Vault (~/Documents/minime-vault): notes FS               │
│  - SQLite DB (./data/minime.db): embeddings, graph, metadata         │
│  - Config YAML (./config/identity.yaml, agents.yaml): identity,      │
│    agents, masks, provider routing                                   │
│  - Traces JSONL (./logs/traces.jsonl): all runs                      │
│  - Feedback JSONL (./logs/feedback.jsonl): user corrections/deltas   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

1. User submits task → Classifier infers domain/complexity
2. Mask system generates hierarchical weights (Global + Domain + Task + Agent)
3. Context Manager retrieves memory using graph-aware, masked retrieval
   - If vault empty (cold start): returns empty results, flags `is_cold_start=True`
   - If vault has content: normal retrieval with graph traversal
4. Context Broker assembles (weighted blocks + token budget)
   - Always includes: Identity principles (P_global)
   - If cold start: General knowledge fallback instead of memory
   - If memory available: Retrieved chunks from vault
   - Always includes: Task instructions
5. LLM generation (frozen, structured output)
6. Action proposals → risk assessment → execution (risk-based gating)
7. Traces logged; feedback collected → offline training loop

---

## 2) MVP Scope (2–3 weeks)

### Features (Explicit)

- ✅ **Task Classification**: Detect domain from user input; route to agent team
- ✅ **Identity Layer**: Store P_global (embedding vectors) with magnitude, decay, scope tags
- ✅ **Obsidian Vault Integration**:
  - VaultIndexer watches Obsidian filesystem (~/obsidian-vault or config path)
  - Extracts frontmatter, wikilinks, tags
  - Computes embeddings per note/chunk
  - Stores metadata in SQLite
  - **Works with empty vault**: System functions from day 1, improves as vault grows
- ✅ **Graph Store & Proposal System**:
  - VaultNode model (path, title, frontmatter, links, tags, timestamps)
  - GraphEdge model (source, target, type, weight, rationale, created_at)
  - Context Manager proposes edges based on similarity + metadata heuristics
  - Proposal review (CLI: `minime graph list-proposals` → approve/reject)
  - Only approved edges are stored
- ✅ **Mask System (Heuristic First)**:
  - Global/Domain/Task/Agent hierarchy
  - Heuristic weights (α, β, γ hardcoded; MLP stub for future)
  - Retrieval weighting: k, temperature, rigor knobs
  - Agent routing bias
- ✅ **Context Manager & Retrieval**:
  - Graph-aware retrieval: vector search + graph traversal + weighting
  - Filters: domain, scope, tags, recency
  - Respects PII/privacy gating (no cross-project leak)
  - Token budget enforcement (context broker)
  - **Cold start support**: Gracefully handles empty vault (returns empty results, flags `is_cold_start`)
  - Fallback to general knowledge when no vault memory available
- ✅ **Multi-Agent Orchestration**:
  - 5 agents (Architect, Research, Builder, Critic, Integrator)
  - Shared context broker
  - Critic enforces constraints; Integrator merges outputs
  - Simple loop: plan → research → build → critique → integrate
- ✅ **Provider Routing Interface**:
  - ProviderInterface abstract class
  - One concrete implementation (OpenAI-compatible)
  - Router selects provider per agent/task
  - Structured output (JSON schema) support
  - Tool calling support
- ✅ **Action Proposal & Risk-Based Gating**:
  - Agents propose actions (write_file, apply_patch, run_command)
  - ActionProposal schema with risk assessment
  - Risk-based execution: safe actions auto-execute, low-risk show preview, high-risk require approval
  - Tool executor with allowlist (safety tags) + path allowlisting
- ✅ **Tracing**:
  - JSONL trace file (./logs/traces.jsonl)
  - Every event: retrievals, masks applied, agent outputs, decisions, token budgets
  - Timestamp, agent, event_type, payload
- ✅ **CLI**:
  - `minime init`: setup vault, DB, config
  - `minime task <query>`: run task (orchestrator end-to-end)
  - `minime index`: refresh vault embeddings (offline)
  - `minime graph list-proposals`: review graph proposals
  - `minime graph approve <edge_id>`: approve edge
  - `minime feedback add <correction_json>`: store feedback
  - `minime trace view [--last N]`: view latest traces
  - `minime config`: show identity, agents, masks
- ✅ **Feedback Integration**:
  - FeedbackCorrectionPair schema
  - Store in feedback.jsonl (delta + scope + rationale)
  - Offline loop reads feedback, updates P_global vectors (stub)

### Non-Features (Explicit)

- ❌ No fine-tuning of LLMs
- ❌ No fully autonomous execution (risk-based gating: safe actions auto-execute, high-risk require approval)
- ❌ No full LoRA adapters (stubs only; v2)
- ❌ No GUI (CLI only for MVP; VS Code integration v2)
- ❌ No sandbox execution (tool allowlist only; v2)
- ❌ No multi-user support
- ❌ No cloud sync (local-first only)
- ❌ No voice/multimodal (text only)
- ❌ No structured reasoning output ranking (v2)
- ❌ No per-project profiles (global only for MVP; v2)

### Acceptance Criteria Checklist

**Core Functionality**
- [ ] `minime init` creates vault dir, SQLite DB, config YAML with no errors
- [ ] VaultIndexer detects changes to Obsidian vault (new notes, edits) and reindexes
- [ ] Graph proposals are generated for new/edited notes (similarity + metadata heuristics)
- [ ] User can approve/reject graph proposals via CLI
- [ ] Context Manager retrieves notes by query (vector + graph traversal + filters)
- [ ] Retrieval respects domain/scope tags (no PII leakage between projects)
- [ ] Mask system generates hierarchical weights; retrieval uses them
- [ ] Provider router selects an LLM provider; generation works
- [ ] Agents can be created/queried; orchestrator loops through them
- [ ] Critic reviews outputs; flags constraint violations
- [ ] Integrator merges agent outputs into final result
- [ ] Actions are risk-assessed; safe actions auto-execute, high-risk require approval
- [ ] Tools execute only if allowlist tag matches (dry-run + execute modes)
- [ ] Every run produces JSONL trace with retrievals, masks, agent outputs, budgets
- [ ] Feedback can be stored; traces show feedback signals
- [ ] `minime task "sample task"` completes end-to-end without crashing

**Data Integrity**
- [ ] Embeddings stored in SQLite match recomputed embeddings (deterministic)
- [ ] Graph edges persist across runs
- [ ] Identity vectors (P_global) persist in config and can be reloaded
- [ ] Traces are append-only JSONL (no corruption on concurrent writes)
- [ ] Feedback JSONL is valid JSON per line

**Security/Safety**
- [ ] Tool executor rejects commands not in allowlist (by tag)
- [ ] Cross-project memory retrieval is blocked (domain/scope filters)
- [ ] Graph proposals flag PII-like content (heuristic check)
- [ ] No credentials or secrets logged in traces
- [ ] High-risk operations require approval; safe operations auto-execute with logging

**Clarity**
- [ ] CLI commands have help text (`--help`)
- [ ] Error messages indicate root cause (missing vault, DB error, etc.)
- [ ] Traces are human-readable JSON (pretty-printed)
- [ ] Config YAML is documented (comments)

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **Slow embeddings** (large vault) | Chunk notes; batch embed; cache in SQLite; warn on large vaults in MVP |
| **Noisy graph proposals** | Similarity threshold + metadata heuristic filter; require user approval |
| **Token budget exceeded** | Context broker truncates/summarizes; warns user; retries with tighter budget |
| **LLM API failures** | Retry logic; timeout; fallback to stub response (trace shows failure) |
| **Agent loops forever** | Hard iteration limit (10); termination check (Critic score threshold) |
| **Cross-project data leakage** | Scope/domain tags mandatory; retrieve query filters enforce scope |
| **Feedback → divergent behavior** | Low learning rate (decay); user approval on P_global updates; reviewable deltas |
| **Trace file grows unbounded** | Rotated logs; archival; configurable retention |

---

## 3) Tech Stack + Why

### Language & Core Frameworks

- **Language**: Python 3.11+
  - Rich ecosystem (typing, async, YAML parsing, SQLite, async requests)
  - Easy ML integration (numpy, scipy, embeddings libraries)
  - Matches your CS/biotech stack
  
- **CLI Framework**: Typer (not Click)
  - Type-safe CLI generation
  - Auto `--help`
  - Async-friendly
  
- **Config**: Pydantic v2 + YAML
  - Strong validation
  - IDE autocomplete
  - Schema serialization (JSON schema for tools)
  
- **HTTP Client**: `httpx` (async, timeouts, retries)
  
- **Async Runtime**: `asyncio` + `anyio` for cancellation safety

### Vector DB & Storage

- **Embeddings Storage**: SQLite with JSON columns
  - Local-first (no external service)
  - Portable (single .db file)
  - Queryable (SQL for metadata filtering)
  - Path to production: migrate to pgvector (PostgreSQL) or Milvus, no code change (interface abstraction)
  
- **Embeddings Library**: `sentence-transformers` (MiniLM-L6-v2 by default)
  - Lightweight (384-dim, ~22M params)
  - Fast (~300 docs/sec on CPU)
  - Deterministic (same text = same embedding)
  - Can swap to OpenAI embeddings via interface
  
- **Chunking Strategy**: 
  - Per-note atomic chunks (frontmatter + body)
  - Overlap chunking for long notes (512 tokens, 128 overlap)
  - Metadata tags preserved per chunk

### Trace & Metadata Storage

- **Traces**: JSONL file (append-only, streaming-friendly)
  - One event per line
  - Timestamp, agent, event_type, payload
  - Tools: `jq` for CLI querying; Python for analysis
  
- **Feedback**: Separate JSONL (corrections, deltas, scope)

- **Config**: YAML files in `./config/`
  - `identity.yaml`: P_global vectors, decay, scope
  - `agents.yaml`: AgentSpec definitions, tool allowlists
  - `masks.yaml`: heuristic weights (α, β, γ) per domain

### Job Runner (Offline Updates)

- **Embedding Refresh**: Scheduled task (`APScheduler` or systemd timer)
  - Detect vault changes → recompute embeddings → update SQLite
  - Runs daily or on manual trigger
  
- **Mask Training**: Background task
  - Reads feedback.jsonl
  - Updates P_global vectors (low-rank updates)
  - Optional: trains MLP stub (PyTorch, 2–3 layers)
  - Runs after N feedback points collected

- **Graph Proposal Generation**: Background task
  - Recompute similarity scores
  - Propose new edges based on heuristics
  - Store in proposals table; wait for approval

### Local-First Dev Mode vs Production Mode

| Aspect | Dev Mode | Production Mode |
|--------|----------|-----------------|
| **Embeddings** | Local (sentence-transformers CPU) | OpenAI API / batch service |
| **DB** | SQLite (./minime.db) | PostgreSQL + pgvector |
| **LLM** | Local (Ollama) or API (gated by `OPENAI_API_KEY`) | API only |
| **Vault** | Local filesystem (~/obsidian-vault) | Synced vault (e.g., iCloud) |
| **Traces** | Local JSONL | Centralized logging (e.g., S3 + CloudWatch) |
| **Config** | Local YAML | Versioned config service |

**Dev Setup** (this week):
```bash
# All local, no API keys required (stub embeddings / mock LLM)
minime init --mode=dev
minime task "write a hello world" --provider=mock
```

**Production Setup** (week 2+):
```bash
# Real embeddings + API
minime init --mode=prod --vault-path=/cloud/obsidian --embedding-provider=openai
minime task "..." --provider=openai  # real API call
```

---

## 4) API + Tooling Plan

### LLM API Usage

**Provider Interface Signature** (Python):
```python
class LLMProvider(ABC):
    async def generate(
        self,
        prompt: str,
        system: str,
        tools: List[ToolDefinition],  # JSON schema
        model: str,
        max_tokens: int,
        temperature: float,
        structured_output_schema: Optional[Dict] = None,
    ) -> GenerationResult:
        """
        Returns: GenerationResult(
            text: str,
            tool_calls: List[ToolCall],
            structured: Optional[Dict],
            stop_reason: str,  # "end_turn" | "tool_use" | "max_tokens"
        )
        """
```

**Concrete Implementation**: OpenAI-compatible (GPT-4 / Claude 3.5)
```python
class OpenAIProvider(LLMProvider):
    async def generate(...):
        # POST to https://api.openai.com/v1/chat/completions
        # Supports vision (via image_url if needed)
        # Supports function_calling (tools)
        # Supports structured output (gpt-4-turbo with JSON schema)
```

**Stub Implementation** (dev/testing):
```python
class MockProvider(LLMProvider):
    async def generate(...):
        # Returns hardcoded output for testing
```

**Usage in Orchestrator**:
```python
provider = router.select_provider(agent=architect_agent, task_domain="architecture")
result = await provider.generate(
    prompt=assembled_context,
    system=agent.system_prompt,
    tools=agent.tools,
    model=agent.model,
    structured_output_schema=agent.output_schema,
)
```

### Embeddings API

**Library**: `sentence-transformers`
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast
embeddings = model.encode(texts, show_progress_bar=False)  # batch encode
```

**Alternative (prod)**: OpenAI embeddings
```python
response = await openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
```

**Chunking Strategy**:
```python
def chunk_note(note_text: str, max_tokens: int = 512, overlap: int = 128) -> List[str]:
    """
    Split note into overlapping chunks.
    Preserve frontmatter as metadata, not content.
    """
    tokens = note_text.split()
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = " ".join(tokens[i : i + max_tokens])
        chunks.append(chunk)
    return chunks
```

### Tool Definitions (JSON Schema)

Every tool is a typed interface:

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    required: List[str]
    allowlist_tags: List[str]  # safety gating
    default_risk_level: str  # "safe" | "low_risk" | "medium_risk" | "high_risk"

# Example tools:

retrieve_memory = ToolDefinition(
    name="retrieve_memory",
    description="Retrieve relevant notes from vault by query",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "filters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "max_age_days": {"type": "integer"},
            },
        },
        "k": {"type": "integer", "description": "Number of results", "default": 5},
    },
    required=["query"],
    allowlist_tags=["read", "memory"],
    default_risk_level="safe",  # read-only, always safe
)

write_file = ToolDefinition(
    name="write_file",
    description="Write or create a file",
    parameters={
        "path": {"type": "string", "description": "File path"},
        "content": {"type": "string", "description": "File content"},
    },
    required=["path", "content"],
    allowlist_tags=["write", "filesystem"],
    default_risk_level="medium_risk",  # risk assessed dynamically based on path
)

apply_patch = ToolDefinition(
    name="apply_patch",
    description="Apply a unified diff patch to a file",
    parameters={
        "path": {"type": "string"},
        "patch": {"type": "string", "description": "Unified diff"},
    },
    required=["path", "patch"],
    allowlist_tags=["write", "filesystem"],
    default_risk_level="medium_risk",  # risk assessed dynamically based on path
)

run_command = ToolDefinition(
    name="run_command",
    description="Execute a shell command",
    parameters={
        "cmd": {"type": "string", "description": "Command to run"},
        "cwd": {"type": "string", "description": "Working directory"},
        "timeout_sec": {"type": "number", "default": 30},
    },
    required=["cmd"],
    allowlist_tags=["execute", "shell"],
    default_risk_level="high_risk",  # commands are high-risk by default
)

search_local_repo = ToolDefinition(
    name="search_local_repo",
    description="Search files by pattern",
    parameters={
        "pattern": {"type": "string", "description": "Regex or glob"},
        "root": {"type": "string", "description": "Root directory"},
    },
    required=["pattern"],
    allowlist_tags=["read", "filesystem"],
    default_risk_level="safe",  # read-only
)

propose_plan = ToolDefinition(
    name="propose_plan",
    description="Propose an architectural plan (structured JSON)",
    parameters={
        "plan": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "modules": {"type": "array"},
                "dependencies": {"type": "array"},
                "phases": {"type": "array"},
            },
        }
    },
    required=["plan"],
    allowlist_tags=["planning"],
    default_risk_level="safe",  # output only, no side effects
)
```

### Tool Allowlist Design & Risk-Based Execution

**Execution Safety**:
1. Agent can only call tools with matching `allowlist_tags`
2. Risk assessment: each action gets a risk level (safe/low/medium/high)
3. **Safe actions**: Auto-execute silently, log to trace
4. **Low-risk actions**: Show preview, auto-execute after 2s (or `--yes` flag)
5. **Medium-risk actions**: Show preview + require one-click approval
6. **High-risk actions**: Block and wait for explicit `y/n` approval
7. Command runner has timeout + output limit
8. File operations respect allowed paths (configurable safe zones)

**Risk Assessment Logic**:
```python
class ActionRiskLevel(str, Enum):
    SAFE = "safe"           # Auto-execute (read-only, output-only)
    LOW_RISK = "low_risk"   # Auto-execute with preview (safe paths)
    MEDIUM_RISK = "medium"  # Preview + one-click approve
    HIGH_RISK = "high"      # Full approval required

def assess_risk(action: ActionProposal, config: Config) -> ActionRiskLevel:
    """Determine risk level based on action type + context."""
    tool_def = get_tool_definition(action.action_type)
    base_risk = tool_def.default_risk_level
    
    # Override based on context
    if action.action_type == "write_file":
        if is_safe_path(action.path, config.safe_paths):
            return ActionRiskLevel.LOW_RISK  # Safe zones = low risk
        elif is_system_path(action.path):
            return ActionRiskLevel.HIGH_RISK  # System paths = high risk
        else:
            return ActionRiskLevel.MEDIUM_RISK
    
    if action.action_type == "run_command":
        if is_allowlisted_command(action.content_or_command, config):
            return ActionRiskLevel.MEDIUM_RISK
        else:
            return ActionRiskLevel.HIGH_RISK
    
    return ActionRiskLevel(base_risk)
```

```python
class ToolExecutor:
    def __init__(self, agent_allowlist_tags: List[str], config: Config):
        self.agent_allowlist_tags = agent_allowlist_tags
        self.config = config  # allowed_paths, safe_paths, etc.
    
    async def execute_tool(
        self,
        tool_call: ToolCall,
        auto_approve_safe: bool = True,
    ) -> ToolResult:
        # 1. Check allowlist
        tool_def = self.registry[tool_call.name]
        if not any(tag in self.agent_allowlist_tags for tag in tool_def.allowlist_tags):
            raise PermissionError(f"Agent cannot call {tool_call.name}")
        
        # 2. Validate arguments against schema
        # 3. Assess risk
        action = ActionProposal.from_tool_call(tool_call)
        risk_level = assess_risk(action, self.config)
        action.risk_level = risk_level
        
        # 4. Handle based on risk
        if risk_level == ActionRiskLevel.SAFE:
            # Auto-execute
            return await self._execute_safe(action)
        elif risk_level == ActionRiskLevel.LOW_RISK:
            # Show preview, auto-execute after delay
            preview = await self._preview_action(action)
            if auto_approve_safe:
                await asyncio.sleep(2)  # Brief pause for user to see
                return await self._execute_action(action)
            else:
                return ToolResult(success=False, output=preview, requires_approval=True)
        elif risk_level == ActionRiskLevel.MEDIUM_RISK:
            # Show preview, wait for approval
            preview = await self._preview_action(action)
            approved = await self._request_approval(action, preview)
            if approved:
                return await self._execute_action(action)
            else:
                return ToolResult(success=False, output="Action cancelled by user")
        else:  # HIGH_RISK
            # Block and require explicit approval
            preview = await self._preview_action(action)
            approved = await self._request_explicit_approval(action, preview)
            if approved:
                return await self._execute_action(action)
            else:
                return ToolResult(success=False, output="Action cancelled by user")
        
        # 5. Trace the tool call
```

### Obsidian Graph Update Flow

**Internal operation** (not exposed to agents):

```python
async def update_obsidian_graph(
    vault_path: str,
    db: AsyncDatabase,
) -> GraphUpdateProposal:
    """
    1. Watch vault for changes (new/edited notes)
    2. Extract frontmatter, wikilinks, tags
    3. Compute embeddings (batch)
    4. Store new nodes in DB
    5. Run similarity search → propose edges
    6. Heuristic filters (metadata match, confidence threshold)
    7. Store proposals in proposals table
    8. Return proposals (user can approve/reject via CLI)
    9. On approval, store edges in graph_edges table
    """
    # Pseudo-code
    vault_state = await load_vault_state(vault_path)
    db_state = await db.load_node_metadata()
    
    new_notes = vault_state - db_state
    for note in new_notes:
        chunks = chunk_note(note.content)
        embeddings = model.encode(chunks)
        # Store in DB
        node_id = await db.insert_node(note.path, note.title, embeddings[0])
        
        # Propose edges
        similar_nodes = await db.semantic_search(embeddings[0], k=5)
        for node_id_other, similarity, metadata in similar_nodes:
            if similarity > 0.7 and matches_heuristic(metadata):
                # Auto-approve high-confidence edges (>0.9), require approval for lower confidence
                proposal = GraphUpdateProposal(
                    source=node_id,
                    target=node_id_other,
                    type="similar",
                    weight=similarity,
                    rationale="semantic similarity + shared domain tags",
                    confidence=similarity,
                    requires_user_approval=similarity < 0.9,  # Auto-approve high-confidence (>0.9)
                )
                await db.store_proposal(proposal)
    
    return proposals
```

---

## 5) Core Data Schemas (Pydantic Models)

```python
# file: minime/schemas.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np

# ============================================================================
# Identity Layer
# ============================================================================

class IdentityPrinciple(BaseModel):
    """A single principle in the identity matrix P_global."""
    id: str
    name: str
    description: str
    vector: List[float]  # embedding
    magnitude: float  # importance weight
    decay_rate: float  # how fast it adapts (0.0 = frozen, 1.0 = plastic)
    scope: str  # "global" | "domain:biotech" | "task:protein_design"
    tags: List[str]  # ["architecture", "rigor", "clarity"]
    created_at: datetime
    updated_at: datetime

class GlobalIdentityMatrix(BaseModel):
    """P_global: all identity principles."""
    principles: List[IdentityPrinciple]
    version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {p.id: p.vector for p in self.principles}

# ============================================================================
# Memory Layer (Obsidian Graph)
# ============================================================================

class VaultNode(BaseModel):
    """One note in Obsidian vault."""
    node_id: str  # unique ID (hash of path)
    path: str  # relative path in vault
    title: str
    frontmatter: Dict[str, Any]  # parsed YAML front matter
    tags: List[str]  # from frontmatter + inline tags
    domain: Optional[str]  # extracted or set by user
    scope: str  # "global" | "project:myproj"
    links: List[str]  # wikilinks [[...]]
    backlinks: List[str]  # incoming links
    created_at: datetime
    updated_at: datetime
    embedding_ref: Optional[str]  # pointer to embeddings table

class GraphEdge(BaseModel):
    """Explicit or proposed edge in the graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # "wikilink" | "backlink" | "similar" | "related"
    weight: float  # 0.0 to 1.0
    rationale: str  # why this edge exists
    confidence: float  # 0.0 to 1.0 (for proposed edges)
    is_approved: bool
    created_at: datetime
    approved_at: Optional[datetime]

class GraphUpdateProposal(BaseModel):
    """Proposal to add/remove edges."""
    proposal_id: str
    edges_to_add: List[GraphEdge]
    edges_to_remove: List[str]  # edge IDs
    confidence: float
    requires_user_approval: bool
    rationale: str
    created_at: datetime

class MemoryChunk(BaseModel):
    """One chunk of a note (for retrieval)."""
    chunk_id: str
    node_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]  # domain, tags, scope, etc.
    position: int  # byte offset in note

# ============================================================================
# Retrieval
# ============================================================================

class RetrievalQuery(BaseModel):
    """A memory retrieval request."""
    query: str
    query_embedding: Optional[List[float]] = None
    filters: Optional[Dict[str, Any]] = None  # domain, tags, scope, max_age_days
    k: int = 5
    include_graph_neighbors: bool = True
    max_context_tokens: int = 2000

class RetrievalResult(BaseModel):
    """Result of a memory retrieval."""
    chunks: List[MemoryChunk]
    scores: List[float]  # similarity + graph + recency weighting
    graph_traversal_depth: int
    blocked_count: int  # chunks blocked by scope/security
    context_tokens_used: int
    is_cold_start: bool = False  # True if vault is empty, no memory available

# ============================================================================
# Masking System
# ============================================================================

class MaskWeights(BaseModel):
    """Output of mask network: controls retrieval, generation, routing."""
    # Retrieval
    retrieval_k: int  # how many results to fetch
    retrieval_min_similarity: float  # threshold
    block_weights: Dict[str, float]  # per-block weights in context
    
    # Generation
    temperature: float
    verbosity: int  # 1=terse, 5=verbose
    rigor: int  # 1=loose, 5=strict
    
    # Agent routing
    agent_routing_bias: Dict[str, float]  # {"architect": 0.8, "builder": 0.2, ...}
    
    # Graph
    graph_proximity_weight: float
    graph_max_hops: int
    
    # Applied masks
    global_mask_strength: float
    domain_mask_strength: float
    task_mask_strength: float
    agent_mask_strength: float

class MaskNetworkInput(BaseModel):
    """Input to mask network MLP."""
    z_identity: List[float]  # P_global embedding
    z_task: List[float]  # current task embedding
    z_domain: List[float]  # domain embedding
    agent_type: str

class MaskNetworkOutput(BaseModel):
    """Output of mask network."""
    weights: MaskWeights
    debug_info: Dict[str, Any]

# ============================================================================
# Agent System
# ============================================================================

class AgentMessage(BaseModel):
    """One message from an agent."""
    agent_name: str
    role: str  # "system" | "assistant" | "user" | "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentSpec(BaseModel):
    """Definition of an agent (can be persisted in config)."""
    name: str
    purpose: str
    io_schema: Dict[str, Any]  # input schema + output schema
    system_prompt: str
    model: str  # "gpt-4" | "claude-3-sonnet" | etc.
    tools_allowed: List[str]  # tool names
    routing_hints: Optional[Dict[str, float]] = None  # when to prefer this agent
    constraints: List[str]  # what this agent must NOT do
    temperature: float = 0.7
    max_tokens: int = 2000
    created_at: datetime
    approved_at: Optional[datetime]

class ToolDefinition(BaseModel):
    """Definition of a tool agents can call."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    required: List[str]
    allowlist_tags: List[str]  # ["read", "write", "execute"]
    requires_approval: bool = True

class ToolCall(BaseModel):
    """One tool call from an agent."""
    tool_name: str
    arguments: Dict[str, Any]

class ToolResult(BaseModel):
    """Result of tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# ============================================================================
# Actions & Approval
# ============================================================================

class ActionProposal(BaseModel):
    """Proposal for an action (write/run/patch)."""
    action_id: str
    action_type: str  # "write_file" | "apply_patch" | "run_command"
    path: str
    content_or_command: str
    dry_run_output: Optional[str] = None  # what would happen (preview)
    justification: str
    risk_level: str  # "safe" | "low_risk" | "medium_risk" | "high_risk"
    created_at: datetime
    approved_at: Optional[datetime]
    executed_at: Optional[datetime]

# ============================================================================
# Tracing & Feedback
# ============================================================================

class TraceEvent(BaseModel):
    """One event in a trace."""
    event_id: str
    timestamp: datetime
    event_type: str  # "retrieve" | "mask_apply" | "agent_output" | "action_propose" | "tool_execute"
    agent_name: Optional[str]
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class RunTrace(BaseModel):
    """Complete trace of one run."""
    run_id: str
    task_query: str
    domain: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    events: List[TraceEvent]
    token_usage: Dict[str, int]  # {"prompt": 100, "completion": 50}
    final_output: Optional[str]
    success: bool

class FeedbackCorrectionPair(BaseModel):
    """User feedback for learning."""
    feedback_id: str
    run_id: str
    original_output: str
    corrected_output: Optional[str] = None
    delta: Optional[str] = None  # diff or edit
    feedback_type: str  # "accept" | "reject" | "edit"
    scope: str  # "global" | "domain" | "task"
    rationale: str
    created_at: datetime

# ============================================================================
# Configuration
# ============================================================================

class MiniMeConfig(BaseModel):
    """Top-level config."""
    vault_path: str  # path to Obsidian vault
    db_path: str  # path to SQLite
    embedding_model: str  # "all-MiniLM-L6-v2" | "openai"
    embedding_cache_size: int  # MB
    default_provider: str  # "openai" | "mock" | "anthropic"
    trace_dir: str  # ./logs/
    config_dir: str  # ./config/
    max_context_tokens: int  # per-generation limit
    enable_offline_training: bool
    offline_training_interval_hours: int
    # Risk-based execution settings
    safe_paths: List[str]  # paths where writes are low-risk (e.g., ["./outputs/", "./docs/"])
    system_paths: List[str]  # paths that are always high-risk (e.g., ["/usr/", "/etc/"])
    allowlisted_commands: List[str]  # commands that are medium-risk (e.g., ["git status", "ls"])
    auto_approve_safe: bool = True  # auto-execute safe actions
    low_risk_auto_delay_sec: float = 2.0  # delay before auto-executing low-risk actions
```

---

## 6) Mask System Implementation Plan

### Mask Hierarchy

```
        P_global (Identity)
             │
    ┌────────┼────────┐
    ▼        ▼        ▼        ▼
  Global   Domain   Task    Agent
  Mask     Mask    Mask    Mask
  
  α=1.0    β=0.3   γ=0.2   γ_agent=0.15
  
  ──────────────────────────────────────────
  P_effective = P_global 
              + α·P_domain     (if domain detected)
              + β·P_task       (for this task)
              + γ·P_agent      (per agent role)
  
  → Outputs: MaskWeights (retrieval_k, temperature, rigor, routing_bias, ...)
```

### How α, β, γ Are Selected (MVP: Heuristic)

```python
def compute_mask_weights(
    task_domain: Optional[str],
    task_complexity: str,  # "simple" | "medium" | "complex"
    agent_type: str,  # "architect" | "builder" | "critic" | ...
) -> MaskWeights:
    """
    MVP: hardcoded heuristics.
    v2: trained MLP (z_identity, z_task, z_domain) → weights
    """
    
    # Heuristic: start with global
    alpha = 1.0  # global always active
    beta = 0.0   # domain multiplier
    gamma = 0.0  # task multiplier
    gamma_agent = 0.0  # agent multiplier
    
    # Domain adjust
    if task_domain:
        beta = 0.3  # domain principles matter somewhat
    
    # Complexity adjust
    if task_complexity == "complex":
        beta += 0.2  # pull more domain knowledge
        gamma = 0.2  # task-specific masking needed
    
    # Agent adjust (always some agent context)
    gamma_agent = {
        "architect": 0.15,  # architect sees structural principles
        "builder": 0.10,    # builder sees execution principles
        "critic": 0.20,     # critic sees rigor principles
        "researcher": 0.12,
        "integrator": 0.08,
    }.get(agent_type, 0.10)
    
    # Retrieval
    retrieval_k = 5 + (3 if task_complexity == "complex" else 0)
    
    # Generation
    temperature = 0.7
    rigor = 3 + (2 if agent_type == "critic" else 0)
    verbosity = 3
    
    # Routing: emphasize this agent
    agent_routing_bias = {
        a: (0.5 if a == agent_type else 0.1)
        for a in ["architect", "builder", "critic", "researcher", "integrator"]
    }
    normalize_dict(agent_routing_bias)
    
    return MaskWeights(
        retrieval_k=retrieval_k,
        retrieval_min_similarity=0.5,
        block_weights={
            "identity": alpha,
            "domain": beta,
            "task": gamma,
            "agent": gamma_agent,
        },
        temperature=temperature,
        verbosity=verbosity,
        rigor=rigor,
        agent_routing_bias=agent_routing_bias,
        global_mask_strength=alpha,
        domain_mask_strength=beta,
        task_mask_strength=gamma,
        agent_mask_strength=gamma_agent,
        graph_proximity_weight=0.2,  # slight boost for neighbors
        graph_max_hops=2,
    )
```

### Temporary Context Abstraction (Subspace Projection)

```python
async def apply_temporary_context_abstraction(
    query_embedding: List[float],
    relevant_node_ids: List[str],
    context_manager,
) -> Dict[str, Any]:
    """
    Create a temporary subspace for a task:
    1. Extract vectors for relevant nodes only
    2. Renormalize within subspace (PCA, ICA, etc.)
    3. Compute new retrieval weights in subspace
    4. Generate using subspace vectors
    5. Discard or merge back to global
    """
    # Step 1: Extract subspace vectors
    subspace_vectors = await context_manager.retrieve_vectors(relevant_node_ids)
    
    # Step 2: Compute PCA basis (low-rank, e.g., k=10)
    pca = PCA(n_components=10)
    subspace_basis = pca.fit_transform(subspace_vectors)
    
    # Step 3: Project query into subspace
    query_projected = pca.transform([query_embedding])[0]
    
    # Step 4: Recompute similarity + weights within subspace
    similarities = cosine_similarity([query_projected], subspace_basis)[0]
    
    # Step 5: Assemble context using subspace weights (lower variance)
    subspace_context = assemble_context_weighted(
        vectors=subspace_basis,
        weights=softmax(similarities),
        max_tokens=1500,
    )
    
    # Step 6: Optional—merge changes back (for learning)
    # For MVP: discard (no global update)
    
    return {
        "subspace_context": subspace_context,
        "subspace_vectors": subspace_basis,
        "projected_query": query_projected,
        "is_temporary": True,
    }
```

### Initial Heuristic + Trainable MLP (Path Forward)

**MVP (Heuristic Only)**:
```python
class MaskSystemMVP:
    """Heuristic-only masking."""
    
    async def compute_weights(
        self,
        task_domain: Optional[str],
        task_complexity: str,
        agent_type: str,
    ) -> MaskWeights:
        return compute_mask_weights(task_domain, task_complexity, agent_type)
```

**v2 (Trainable MLP)**:
```python
class MaskNetworkMLP(nn.Module):
    """Small trainable network."""
    
    def __init__(self, input_dim: int = 1152, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            # Output: α, β, γ (3 values)
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),  # output in [0, 1]
        )
    
    def forward(self, z_identity, z_task, z_domain):
        x = torch.cat([z_identity, z_task, z_domain], dim=-1)
        alpha_beta_gamma = self.net(x)
        return alpha_beta_gamma

# Training loop (offline)
async def train_mask_network(
    feedback_jsonl_path: str,
    device: str = "cpu",
):
    """
    Read feedback.jsonl → (z_identity, z_task, z_domain, labels)
    Train MLP to predict α, β, γ
    Save weights
    """
    model = MaskNetworkMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    data = load_feedback_training_data(feedback_jsonl_path)  # List[MaskTrainingExample]
    
    for epoch in range(10):
        total_loss = 0.0
        for example in data:
            z_identity, z_task, z_domain = example.embeddings
            true_mask_weights = example.true_weights
            
            pred_weights = model(
                torch.tensor(z_identity).to(device),
                torch.tensor(z_task).to(device),
                torch.tensor(z_domain).to(device),
            )
            
            # Loss: MSE between predicted and true α, β, γ
            loss = F.mse_loss(pred_weights, torch.tensor(true_mask_weights).to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: loss={total_loss / len(data):.4f}")
    
    torch.save(model.state_dict(), "./checkpoints/mask_network.pt")
```

### Training Data Format (from Feedback)

```python
class MaskTrainingExample(BaseModel):
    """One training example for mask network."""
    z_identity: List[float]  # embedding of P_global at time of task
    z_task: List[float]  # embedding of task query
    z_domain: List[float]  # embedding of detected domain
    true_mask_weights: Dict[str, float]  # {alpha, beta, gamma}
    # ^ inferred from: was output accepted? if rejected, what changed?
    # ^ low confidence = no update
    
    @classmethod
    def from_feedback_pair(
        cls,
        feedback: FeedbackCorrectionPair,
        run_trace: RunTrace,
        context_manager,
    ):
        """Extract training example from user feedback."""
        # Only use high-confidence feedback
        if feedback.feedback_type == "accept":
            confidence = 0.9  # strong signal
        elif feedback.feedback_type == "edit":
            confidence = 0.5  # weak signal
        else:
            confidence = 0.0  # no learning from rejections
        
        # Compute target mask weights from trace
        true_alpha, true_beta, true_gamma = infer_weights_from_trace(run_trace)
        
        return cls(
            z_identity=...,  # from trace
            z_task=...,      # task embedding
            z_domain=...,    # domain embedding
            true_mask_weights={"alpha": true_alpha, "beta": true_beta, "gamma": true_gamma},
        )
```

### How Graph Proximity + Node Types Influence Retrieval Weights

```python
async def reweight_by_graph_proximity(
    candidate_chunks: List[MemoryChunk],
    query_node_id: Optional[str],  # if known
    context_manager,
    graph_proximity_weight: float = 0.2,
    max_hops: int = 2,
) -> List[Tuple[MemoryChunk, float]]:
    """
    Boost weights for chunks in nearby nodes (graph neighbors).
    """
    reweighted = []
    
    for chunk in candidate_chunks:
        base_score = chunk.similarity  # from vector search
        
        # Graph proximity boost
        if query_node_id:
            distance = await context_manager.graph_distance(
                query_node_id,
                chunk.node_id,
                max_hops=max_hops,
            )
            if distance > 0 and distance <= max_hops:
                # Exponential decay with hops
                proximity_boost = graph_proximity_weight * (1.0 / (1.0 + distance))
                base_score += proximity_boost
        
        # Node type modifier
        node_type = chunk.metadata.get("node_type", "general")
        type_boost = {
            "definition": 0.05,
            "principle": 0.10,
            "decision": 0.08,
            "experiment": 0.03,
            "general": 0.0,
        }.get(node_type, 0.0)
        base_score += type_boost
        
        reweighted.append((chunk, base_score))
    
    # Renormalize
    max_score = max(s for _, s in reweighted)
    normalized = [(c, s / max_score) for c, s in reweighted]
    
    return sorted(normalized, key=lambda x: x[1], reverse=True)
```

### Cold Start Handling (Empty Vault)

**Problem**: Vault starts empty; system must work from day 1 and improve as vault grows.

**Solution**: Graceful degradation with fallback to general knowledge.

```python
class ContextManager:
    """Handles memory retrieval with cold start support."""
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5,
    ) -> RetrievalResult:
        """Retrieve memory, gracefully handle empty vault."""
        
        # Try vector search
        chunks = await self._vector_search(query, k=k, filters=filters)
        
        # Check if vault is empty (cold start)
        if not chunks or len(chunks) == 0:
            # Count total nodes in vault
            total_nodes = await self.db.count_nodes()
            
            if total_nodes == 0:
                # Cold start: vault is empty
                return RetrievalResult(
                    chunks=[],
                    scores=[],
                    graph_traversal_depth=0,
                    blocked_count=0,
                    context_tokens_used=0,
                    is_cold_start=True,  # Flag for orchestrator
                )
        
        # Normal retrieval: vault has content
        # Apply graph traversal, reweighting, etc.
        enriched_chunks = await self._enrich_with_graph(chunks)
        
        return RetrievalResult(
            chunks=enriched_chunks,
            scores=[c.similarity for c in enriched_chunks],
            graph_traversal_depth=1,
            blocked_count=0,
            context_tokens_used=sum(len(c.content) for c in enriched_chunks),
            is_cold_start=False,
        )
```

**Orchestrator Cold Start Handling**:

```python
class Orchestrator:
    async def _assemble_context(
        self,
        task_query: str,
        memory_result: RetrievalResult,
        mask_weights: MaskWeights,
    ) -> List[Tuple[str, str, float]]:
        """Assemble context blocks, handle cold start."""
        
        context_blocks = []
        
        # 1. Identity block (always available)
        identity_text = self._format_identity_principles()
        context_blocks.append(("identity", identity_text, mask_weights.block_weights.get("identity", 1.0)))
        
        # 2. Memory block (if available)
        if memory_result.is_cold_start:
            # Cold start: use general knowledge fallback
            general_knowledge_prompt = """
            No personal memory available yet (vault is empty).
            Use general domain knowledge and best practices.
            Apply the user's identity principles (listed above) to guide your approach.
            """
            context_blocks.append(("general_knowledge", general_knowledge_prompt, 0.6))
            
            # Log cold start for user awareness
            await self.tracer.log_event(
                self.current_run_id,
                "cold_start",
                {"message": "Vault is empty, using general knowledge + identity principles"},
            )
        elif memory_result.chunks:
            # Normal: use retrieved memory
            memory_text = self._format_memory_chunks(memory_result.chunks)
            context_blocks.append(("memory", memory_text, mask_weights.block_weights.get("memory", 0.8)))
        
        # 3. Task instructions
        context_blocks.append(("task", task_query, 1.0))
        
        return context_blocks
```

**User Feedback in CLI**:

```python
# In CLI task command
if memory_result.is_cold_start:
    typer.echo("ℹ️  Vault is empty - using general knowledge + your identity principles")
    typer.echo("   Add notes to your vault to enable personalized memory retrieval")
else:
    typer.echo(f"📚 Found {len(memory_result.chunks)} relevant notes from your vault")
```

**Progressive Enhancement**:
- Day 1: Empty vault → Identity + general knowledge
- Week 1: Some notes → Identity + few notes + general knowledge
- Month 1: Rich vault → Identity + many notes + graph connections + general knowledge

---

## 7) Multi-Agent Orchestration Plan

### Agent Team Composition & Visibility

| Agent | What It Sees | Inputs | Outputs | Loop Role |
|-------|-------------|--------|---------|-----------|
| **Architect** | Global identity, domain principles, task requirements | Masked context, task | System design, module boundaries, dependencies | Plan/design |
| **Research** | Domain knowledge, papers, prior work, related designs | Query, masked retrieval | Relevant knowledge snippets, synthesis | Inform |
| **Builder** | Architecture, code patterns, style guide | Architect's plan, research results | Partial/full implementation, configs | Implement |
| **Critic** | Principles, constraints, style rules | All outputs so far | Violations list, revision requests | Validate |
| **Integrator** | All agent outputs | Architect + Builder + Critic outputs | Merged, conflict-resolved final output | Integrate |

### Orchestrator State Machine

```
USER SUBMITS TASK
  ↓
CLASSIFY (infer domain, complexity)
  ↓
SELECT TEAM (Architect + Research + Builder + Critic + Integrator)
  ↓
GENERATE MASKS (heuristic or MLP)
  ↓
┌─ AGENT LOOP ─────────────────────────────────────────┐
│                                                      │
│  FOR agent in [Architect, Research, Builder, Critic]:│
│    │                                                  │
│    ├─ Retrieve memory (masked)                        │
│    │  └─ If cold start: use general knowledge fallback│
│    ├─ Assemble context (block weights)               │
│    │  └─ Identity + Memory (if available) + Task     │
│    ├─ LLM generation (structured output)             │
│    ├─ Parse structured output                        │
│    ├─ Validate against agent constraints             │
│    ├─ Trace event (retrieval, mask, output)          │
│    └─ Add to shared context                          │
│                                                      │
│  CRITIC REVIEWS:                                     │
│    ├─ Check all outputs for constraint violations    │
│    ├─ Request revisions if needed (loop back)        │
│    └─ Flag warnings (trace)                          │
│                                                      │
│  TERMINATION:                                        │
│    ├─ If Critic approved: exit loop                  │
│    ├─ Else if iteration < 5: loop back (revision)   │
│    ├─ Else: fail + propose simplification            │
│                                                      │
└──────────────────────────────────────────────────────┘
  ↓
INTEGRATOR MERGES:
  ├─ Resolve conflicts (duplication, contradictions)
  ├─ Prioritize based on task
  └─ Produce final output
  ↓
PROPOSE ACTIONS
  ├─ write_file / apply_patch / run_command
  ├─ Assess risk level (safe/low/medium/high)
  ├─ Safe: auto-execute + log
  ├─ Low-risk: preview + auto-execute (2s delay)
  ├─ Medium-risk: preview + one-click approve
  └─ High-risk: block + explicit approval required
  ↓
EXECUTE + TRACE
  └─ Tool executor with allowlist + risk-based gating
  ↓
COLLECT FEEDBACK (explicit or implicit from traces)
  ↓
OFFLINE TRAINING (async)
  └─ Mask network fine-tuning
```

### Critic Constraints Enforcement

```python
class CriticAgent:
    """Enforces principles and constraints."""
    
    def __init__(
        self,
        principles: List[IdentityPrinciple],
        agent_constraints: Dict[str, List[str]],  # per-agent constraints
    ):
        self.principles = principles
        self.agent_constraints = agent_constraints
    
    async def review(
        self,
        architect_output: str,
        builder_output: str,
        research_output: str,
    ) -> CriticReview:
        """
        Ask LLM to review outputs against constraints.
        """
        review_prompt = f"""
You are a technical critic. Review the following outputs against these principles:
{self._format_principles()}

Architect output:
{architect_output}

Builder output:
{builder_output}

Research output:
{research_output}

Check for:
1. Violations of stated principles
2. Inconsistencies between agents
3. Unsafe patterns (e.g., unvalidated user input)
4. Missing error handling
5. Unclear abstractions

Respond in JSON:
{{
  "violations": [
    {{"agent": "builder", "issue": "...", "severity": "high|medium|low"}},
    ...
  ],
  "conflicts": [
    {{"between": ["architect", "builder"], "description": "..."}},
    ...
  ],
  "approved": false,
  "revision_request": "..."
}}
"""
        response = await self.provider.generate(
            prompt=review_prompt,
            structured_output_schema=CriticReview.schema(),
        )
        return CriticReview(**response.structured)
    
    def _format_principles(self) -> str:
        return "\n".join(
            f"- {p.name}: {p.description}"
            for p in self.principles
        )
```

### Integrator Conflict Resolution

```python
class IntegratorAgent:
    """Merges agent outputs; resolves conflicts."""
    
    async def integrate(
        self,
        architect_output: Dict,
        builder_output: str,
        research_output: str,
        critic_review: CriticReview,
    ) -> str:
        """
        Merge outputs into final deliverable.
        """
        if critic_review.violations:
            # Flag but continue (warn user)
            violations_text = "\n".join(
                f"- {v.agent}: {v.issue}"
                for v in critic_review.violations
            )
        
        merge_prompt = f"""
You are an integrator. Merge these outputs into a final deliverable:

Architecture (from Architect):
{architect_output}

Implementation Sketch (from Builder):
{builder_output}

Research Summary (from Researcher):
{research_output}

{f"Critic Warnings:" + violations_text if critic_review.violations else ""}

Merge into one cohesive output. Prioritize clarity, consistency, and safety.
Return as structured JSON with sections: summary, design, code_sketch, research_notes.
"""
        response = await self.provider.generate(
            prompt=merge_prompt,
            structured_output_schema=IntegratorOutput.schema(),
        )
        return response.structured
```

### Provider Routing Per Agent

```python
class ProviderRouter:
    """Route to best provider per agent/task."""
    
    def __init__(self, config: MiniMeConfig):
        self.config = config
        self.providers = {
            "openai": OpenAIProvider(api_key=...),
            "anthropic": AnthropicProvider(api_key=...),
            "mock": MockProvider(),
        }
    
    def select_provider(
        self,
        agent_name: str,
        task_domain: Optional[str],
        task_complexity: str,
    ) -> LLMProvider:
        """
        MVP: simple routing heuristic.
        
        Rules:
        - Architect: always use strongest model (GPT-4)
        - Builder: GPT-4 (code gen)
        - Critic: GPT-4 (strict reasoning)
        - Researcher: Claude (context window)
        - Integrator: GPT-4 (synthesis)
        """
        
        routing_rules = {
            "architect": "openai",  # GPT-4
            "builder": "openai",     # GPT-4
            "critic": "openai",      # GPT-4
            "researcher": "anthropic", # Claude (larger context)
            "integrator": "openai",
        }
        
        provider_name = routing_rules.get(agent_name, self.config.default_provider)
        return self.providers[provider_name]
```

### AgentSpec Proposal Flow (Gated)

```python
class OrchestratorAgentGating:
    """Propose + approve new agent types."""
    
    async def propose_agent_spec(
        self,
        task: str,
        reason: str,
    ) -> AgentSpec:
        """
        If no existing agent fits, propose a new one.
        """
        proposal_prompt = f"""
A task does not fit existing agents:

Task: {task}
Reason: {reason}

Propose a new agent spec (JSON):
{{
  "name": "...",
  "purpose": "...",
  "system_prompt": "...",
  "tools_allowed": [...],
  "constraints": [...],
}}
"""
        response = await self.provider.generate(
            prompt=proposal_prompt,
            structured_output_schema=AgentSpec.schema(),
        )
        proposed_spec = AgentSpec(**response.structured)
        
        # Critic reviews spec
        critic_review = await self.critic.review_agent_spec(proposed_spec)
        if critic_review.violations:
            print(f"Critic flagged violations in proposed spec: {critic_review.violations}")
            return None  # proposal rejected
        
        # Ask user for approval
        print(f"\nProposed Agent Spec:\n{proposed_spec.json(indent=2)}")
        approved = input("Approve this agent? (y/n): ") == "y"
        
        if approved:
            # Store in config
            await self.config_manager.save_agent_spec(proposed_spec)
            return proposed_spec
        else:
            return None
```

---

## 8) Repo Scaffold (Exact Tree)

```
minime/
├── README.md                          # Quick start + overview
├── pyproject.toml                     # Poetry / pip dependencies
├── Makefile                           # Dev commands (init, test, run)
│
├── minime/                            # Main package
│   ├── __init__.py
│   ├── cli.py                         # Typer CLI entry point
│   │
│   ├── identity/
│   │   ├── __init__.py
│   │   ├── principles.py              # IdentityPrinciple, GlobalIdentityMatrix
│   │   └── loader.py                  # Load from YAML config
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vault.py                   # VaultIndexer (watches FS, extracts metadata)
│   │   ├── db.py                      # SQLite queries (nodes, edges, embeddings)
│   │   ├── embeddings.py              # Embedding model (SentenceTransformer / OpenAI)
│   │   └── chunk.py                   # Chunking strategy
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── store.py                   # Graph store (SQLite backend)
│   │   ├── proposal.py                # GraphUpdateProposal generation + storage
│   │   ├── traversal.py               # Graph traversal (k-hop neighbors)
│   │   └── heuristics.py              # Edge proposal heuristics
│   │
│   ├── mask/
│   │   ├── __init__.py
│   │   ├── system.py                  # MaskWeights computation (heuristic)
│   │   ├── heuristic.py               # Hardcoded heuristics (MVP)
│   │   ├── network.py                 # MLP (PyTorch, optional v2)
│   │   └── training.py                # Mask network training loop
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   ├── manager.py                 # ContextManager (retrieval + graph-aware)
│   │   ├── broker.py                  # ContextBroker (assembly + token budget)
│   │   ├── retrieval.py               # RetrievalQuery + RetrievalResult logic
│   │   └── filtering.py               # Domain/scope/tag filtering
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                    # Agent base class
│   │   ├── architect.py               # Architect agent
│   │   ├── builder.py                 # Builder agent
│   │   ├── research.py                # Research agent
│   │   ├── critic.py                  # Critic agent
│   │   ├── integrator.py              # Integrator agent
│   │   └── registry.py                # Agent registration + config loading
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # Main orchestration loop
│   │   ├── classifier.py              # Task classification (domain, complexity)
│   │   └── router.py                  # Agent routing + team selection
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                    # LLMProvider abstract class
│   │   ├── openai_compat.py           # OpenAI-compatible (GPT-4, Claude, etc.)
│   │   ├── mock.py                    # MockProvider (testing)
│   │   ├── router.py                  # Provider selection heuristics
│   │   └── structured_output.py       # JSON schema handling
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                    # ToolDefinition, ToolCall, ToolResult
│   │   ├── executor.py                # ToolExecutor (allowlist + approval)
│   │   ├── builtin.py                 # Built-in tools (retrieve_memory, write_file, etc.)
│   │   └── allowlist.py               # Allowlist enforcement
│   │
│   ├── tracing/
│   │   ├── __init__.py
│   │   ├── tracer.py                  # TraceEvent, RunTrace logging
│   │   └── storage.py                 # JSONL trace storage + querying
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── storage.py                 # FeedbackCorrectionPair storage
│   │   └── loader.py                  # Load feedback for training
│   │
│   ├── schemas.py                     # All Pydantic models (centralized)
│   └── config.py                      # Config loading + validation
│
├── config/
│   ├── identity.yaml                  # P_global: identity principles
│   ├── agents.yaml                    # AgentSpec definitions
│   ├── masks.yaml                     # Heuristic mask weights (α, β, γ)
│   ├── providers.yaml                 # Provider routing rules + credentials
│   └── tools.yaml                     # Tool definitions + allowlists
│
├── data/
│   ├── minime.db                      # SQLite (vault nodes, edges, embeddings)
│   └── obsidian-vault/                # Optional local Obsidian vault (symlink or copy)
│
├── logs/
│   ├── traces.jsonl                   # All run traces (append-only)
│   └── feedback.jsonl                 # User feedback (append-only)
│
├── checkpoints/
│   └── mask_network.pt                # Mask MLP weights (PyTorch)
│
├── tests/
│   ├── __init__.py
│   ├── test_vault_indexer.py          # Unit: VaultIndexer
│   ├── test_context_manager.py        # Unit: retrieval + filtering
│   ├── test_mask_system.py            # Unit: mask weights
│   ├── test_tool_executor.py          # Unit: tool allowlist + approval
│   ├── test_graph_proposals.py        # Unit: edge proposal generation
│   ├── test_agent_integration.py      # Unit: agent output + validation
│   ├── test_trace_logging.py          # Unit: trace correctness
│   ├── test_e2e_mock.py               # E2E: full pipeline with mock LLM
│   └── conftest.py                    # Fixtures (fake vault, mock DB, etc.)
│
├── scripts/
│   ├── init_vault.py                  # Script: initialize vault + DB
│   ├── refresh_embeddings.py          # Script: offline embedding refresh
│   ├── train_masks.py                 # Script: offline mask training
│   ├── review_proposals.py            # Script: interactive proposal review
│   └── benchmark.py                   # Script: perf + token metrics
│
└── .gitignore                         # Ignore DB, logs, checkpoints, .env
```

### One-line descriptions per key file

| File | Purpose |
|------|---------|
| `cli.py` | Typer CLI commands (init, task, index, graph, feedback, trace, config) |
| `identity/principles.py` | Identity vector storage + loading |
| `memory/vault.py` | Watch Obsidian FS, extract notes, parse frontmatter |
| `memory/db.py` | SQLite schema + queries (nodes, edges, embeddings) |
| `graph/store.py` | Graph persistence + queries |
| `graph/proposal.py` | Generate + store edge proposals |
| `mask/system.py` | Compute hierarchical mask weights |
| `context/manager.py` | Orchestrate retrieval (vector + graph + filtering) |
| `context/broker.py` | Assemble context blocks; enforce token budget |
| `agents/base.py` | Common agent interface |
| `orchestrator/orchestrator.py` | Main loop (classify → mask → team → loop → integrate) |
| `providers/base.py` | Abstract LLMProvider interface |
| `providers/openai_compat.py` | Real API integration (OpenAI, Anthropic, local) |
| `tools/executor.py` | Execute tool calls; enforce allowlist + approval |
| `tracing/tracer.py` | Log every event (JSONL) |
| `schemas.py` | All Pydantic models (single source of truth) |
| `config.py` | Load + validate config from YAML |

---

## 9) Starter Code (Runnable)

### 9a) CLI Entry Point

**File**: `minime/cli.py`

```python
import asyncio
import typer
from typing import Optional
from pathlib import Path
import json

from minime.orchestrator.orchestrator import Orchestrator
from minime.config import MiniMeConfig
from minime.memory.vault import VaultIndexer
from minime.graph.store import GraphStore
from minime.graph.proposal import GraphProposalManager
from minime.tracing.storage import TraceStorage
from minime.feedback.storage import FeedbackStorage

app = typer.Typer(help="MiniMe: Identity-conditioned LLM orchestration")

# Global state (lazy-loaded)
_config: Optional[MiniMeConfig] = None
_orchestrator: Optional[Orchestrator] = None

def get_config() -> MiniMeConfig:
    global _config
    if _config is None:
        _config = MiniMeConfig.load_from_file("./config/identity.yaml")
    return _config

async def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        config = get_config()
        _orchestrator = Orchestrator(config)
    return _orchestrator

# ============================================================================
# Commands
# ============================================================================

@app.command()
def init(
    vault_path: str = typer.Option(
        "~/obsidian-vault",
        help="Path to Obsidian vault",
    ),
    db_path: str = typer.Option(
        "./data/minime.db",
        help="Path to SQLite DB",
    ),
    mode: str = typer.Option(
        "dev",
        help="'dev' or 'prod'",
    ),
):
    """Initialize MiniMe: vault, DB, config."""
    typer.echo(f"🚀 Initializing MiniMe in {mode} mode...")
    
    vault_path = Path(vault_path).expanduser()
    if not vault_path.exists():
        vault_path.mkdir(parents=True, exist_ok=True)
        typer.echo(f"   Created vault directory: {vault_path}")
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create SQLite schema
    from minime.memory.db import init_db
    init_db(str(db_path))
    typer.echo(f"   Created database: {db_path}")
    
    # Create config directory
    config_dir = Path("./config")
    config_dir.mkdir(exist_ok=True)
    
    # Create default identity.yaml
    identity_yaml = config_dir / "identity.yaml"
    if not identity_yaml.exists():
        default_identity = """
# Default Identity Principles
principles:
  - name: Modularity
    description: "Break systems into independent, testable modules"
    magnitude: 1.0
    decay_rate: 0.05
    scope: global
    tags: [architecture, code]
  
  - name: Clarity Over Cleverness
    description: "Prefer clear, obvious code to clever optimizations"
    magnitude: 0.9
    decay_rate: 0.05
    scope: global
    tags: [code, style]
  
  - name: Reversibility
    description: "Design decisions that can be undone with minimal cost"
    magnitude: 0.8
    decay_rate: 0.05
    scope: global
    tags: [architecture, design]

embedding_model: "all-MiniLM-L6-v2"
default_provider: "mock"
max_context_tokens: 4000

# Risk-based execution settings
safe_paths:
  - "./outputs/"
  - "./docs/"
  - "./tmp/"
system_paths:
  - "/usr/"
  - "/etc/"
  - "/bin/"
allowlisted_commands:
  - "git status"
  - "git log"
  - "ls"
  - "pwd"
auto_approve_safe: true
low_risk_auto_delay_sec: 2.0
"""
        identity_yaml.write_text(default_identity)
        typer.echo(f"   Created config: {identity_yaml}")
    
    typer.echo("✅ MiniMe initialized successfully!")

@app.command()
def task(
    query: str = typer.Argument(..., help="Task description"),
    provider: Optional[str] = typer.Option(
        None,
        help="LLM provider ('openai', 'mock', etc.)",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Propose actions without executing",
    ),
):
    """Run an end-to-end task."""
    typer.echo(f"\n📋 Task: {query}\n")
    
    async def _run():
        orchestrator = await get_orchestrator()
        result = await orchestrator.process_task(
            query=query,
            provider_override=provider,
            dry_run=dry_run,
        )
        
        # Show cold start status if applicable
        if hasattr(result, 'memory_status') and result.memory_status == "cold_start":
            typer.echo("ℹ️  Vault is empty - used general knowledge + your identity principles")
            typer.echo("   Tip: Add notes to your vault for personalized memory retrieval")
        elif hasattr(result, 'memory_chunks_count') and result.memory_chunks_count > 0:
            typer.echo(f"📚 Found {result.memory_chunks_count} relevant notes from your vault")
        
        typer.echo(f"\n✅ Task complete.\n")
        typer.echo(result.final_output)
        
        if result.proposed_actions:
            typer.echo("\n📝 Actions Executed/Proposed:")
            for action in result.proposed_actions:
                risk_emoji = {
                    "safe": "✅",
                    "low_risk": "⚡",
                    "medium_risk": "⚠️",
                    "high_risk": "🛑",
                }.get(action.risk_level, "❓")
                status = "EXECUTED" if action.executed_at else "PENDING APPROVAL"
                typer.echo(f"   {risk_emoji} [{action.action_id}] {action.action_type}: {action.path} ({action.risk_level}) - {status}")
    
    asyncio.run(_run())

@app.command()
def index():
    """Refresh vault embeddings (offline)."""
    typer.echo("🔄 Indexing vault...")
    
    async def _run():
        config = get_config()
        indexer = VaultIndexer(config.vault_path)
        await indexer.index()
        typer.echo(f"✅ Vault indexed.")
    
    asyncio.run(_run())

@app.command()
def graph_proposals():
    """List pending graph update proposals."""
    typer.echo("📊 Graph Proposals:\n")
    
    async def _run():
        config = get_config()
        from minime.memory.db import AsyncDatabase
        
        db = AsyncDatabase(config.db_path)
        proposals = await db.get_pending_proposals()
        
        if not proposals:
            typer.echo("   (no pending proposals)")
            return
        
        for prop in proposals:
            typer.echo(f"[{prop.proposal_id}]")
            typer.echo(f"  Edges to add: {len(prop.edges_to_add)}")
            for edge in prop.edges_to_add:
                typer.echo(f"    - {edge.source_node_id} → {edge.target_node_id}")
                typer.echo(f"      (confidence: {edge.confidence:.2f})")
            typer.echo()
    
    asyncio.run(_run())

@app.command()
def graph_approve(proposal_id: str = typer.Argument(...)):
    """Approve a graph proposal."""
    typer.echo(f"✅ Approving proposal {proposal_id}...")
    
    async def _run():
        config = get_config()
        from minime.memory.db import AsyncDatabase
        
        db = AsyncDatabase(config.db_path)
        await db.approve_proposal(proposal_id)
        typer.echo("   Done.")
    
    asyncio.run(_run())

@app.command()
def feedback_add(
    run_id: str = typer.Argument(...),
    feedback_type: str = typer.Argument(..., help="'accept', 'reject', or 'edit'"),
    delta: Optional[str] = typer.Option(None, help="JSON delta for edits"),
):
    """Add feedback for a run."""
    typer.echo(f"📝 Recording feedback for run {run_id}...")
    
    async def _run():
        config = get_config()
        storage = FeedbackStorage(config)
        
        await storage.store_feedback(
            run_id=run_id,
            feedback_type=feedback_type,
            delta=delta,
        )
        typer.echo("   Feedback stored.")
    
    asyncio.run(_run())

@app.command()
def trace_view(last: int = typer.Option(5, help="Show last N events")):
    """View recent trace events."""
    typer.echo(f"📜 Recent Traces (last {last} events):\n")
    
    config = get_config()
    storage = TraceStorage(config)
    
    events = storage.get_recent_events(n=last)
    for event in events:
        typer.echo(f"[{event.timestamp}] {event.event_type}")
        if event.agent_name:
            typer.echo(f"  Agent: {event.agent_name}")
        typer.echo(f"  {json.dumps(event.payload, indent=2)}")
        typer.echo()

@app.command()
def config_show():
    """Show current configuration."""
    typer.echo("⚙️  Configuration:\n")
    
    config = get_config()
    typer.echo(config.json(indent=2))

if __name__ == "__main__":
    app()
```

### 9b) Local Dev Setup Commands

**File**: `dev-setup.sh`

```bash
#!/bin/bash

# 1. Create virtual env
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Initialize MiniMe
minime init --mode=dev

# 4. Create sample vault notes
mkdir -p data/obsidian-vault
cat > data/obsidian-vault/intro.md << 'EOF'
---
title: Getting Started
domain: general
scope: global
tags: [tutorial]
---

# MiniMe Quick Start

This is a sample note in your Obsidian vault. MiniMe will index it automatically.

## Key Concepts

- Identity: Your principles and preferences
- Memory: Notes and knowledge base
- Masks: Conditional weighting of identity + memory
- Agents: Specialized LLM roles

More at [[further-reading]]
EOF

cat > data/obsidian-vault/further-reading.md << 'EOF'
---
title: Further Reading
tags: [reference]
---

# Additional Resources

- [[intro]]
EOF

# 5. Index vault (optional - works with empty vault too)
minime index  # Will show "Vault is empty" if no notes yet

# 6. Run a test task (works even with empty vault)
minime task "explain the MiniMe architecture" --provider=mock
# → Uses: Identity principles + general knowledge (cold start)
# → As you add notes, system automatically uses them

# 7. View traces
minime trace-view --last=10
```

---

## 10) Testing Strategy

### Unit Tests (pytest)

**File**: `tests/test_vault_indexer.py`

```python
import pytest
from pathlib import Path
from minime.memory.vault import VaultIndexer

@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary Obsidian vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    
    note = vault / "test.md"
    note.write_text("""---
title: Test Note
tags: [test]
domain: general
---

Body of test note.

[[link-to-other]]
""")
    
    return vault

@pytest.mark.asyncio
async def test_vault_indexer_reads_notes(temp_vault):
    """Test that VaultIndexer reads and parses notes."""
    indexer = VaultIndexer(str(temp_vault))
    nodes = await indexer.index()
    
    assert len(nodes) == 1
    node = nodes[0]
    assert node.title == "Test Note"
    assert "test" in node.tags
    assert "link-to-other" in node.links
```

**File**: `tests/test_context_manager.py`

```python
import pytest
from minime.context.manager import ContextManager

@pytest.mark.asyncio
async def test_retrieval_filtering(mock_context_manager):
    """Test that retrieval respects domain filters."""
    # Add test chunks with different domains
    
    result = await mock_context_manager.retrieve(
        query="test query",
        filters={"domain": "biotech"},
        k=5,
    )
    
    # Verify only biotech chunks returned
    assert all(c.metadata["domain"] == "biotech" for c in result.chunks)
```

**File**: `tests/test_mask_system.py`

```python
def test_mask_heuristic():
    """Test mask weight computation."""
    from minime.mask.heuristic import compute_mask_weights
    
    weights = compute_mask_weights(
        task_domain="biotech",
        task_complexity="complex",
        agent_type="architect",
    )
    
    assert weights.retrieval_k >= 5
    assert weights.block_weights["identity"] > 0
    assert weights.block_weights["domain"] > 0
```

**File**: `tests/test_tool_executor.py`

```python
@pytest.mark.asyncio
async def test_tool_allowlist():
    """Test that executor respects allowlist."""
    from minime.tools.executor import ToolExecutor
    from minime.schemas import ToolCall
    
    executor = ToolExecutor(
        agent_allowlist_tags=["read"],  # only read allowed
        config=mock_config,
    )
    
    # Try to execute write tool (should fail)
    result = await executor.execute(
        ToolCall(tool_name="write_file", arguments={"path": "x.txt", "content": "x"}),
        dry_run=False,
    )
    
    assert not result.success
    assert "not allowed" in result.error
```

**File**: `tests/test_graph_proposals.py`

```python
@pytest.mark.asyncio
async def test_graph_proposal_generation():
    """Test that proposals are generated for similar notes."""
    from minime.graph.proposal import GraphProposalManager
    
    manager = GraphProposalManager(mock_graph_store)
    
    # Add two similar notes
    node_a = VaultNode(node_id="a", ...)
    node_b = VaultNode(node_id="b", ...)
    
    proposals = await manager.generate_proposals([node_a, node_b])
    
    assert len(proposals) > 0
    assert any(p.source_node_id == "a" and p.target_node_id == "b" for p in proposals)
```

**File**: `tests/test_trace_logging.py`

```python
def test_trace_logging():
    """Test that traces are logged correctly."""
    from minime.tracing.tracer import Tracer
    
    tracer = Tracer(mock_config)
    run_id = tracer.start_run("test task")
    
    tracer.log_event(run_id, "retrieve", {"num_chunks": 5})
    tracer.log_event(run_id, "mask_apply", {"alpha": 1.0})
    
    trace = tracer.end_run(run_id, "output", True)
    
    assert trace.run_id == run_id
    assert len(trace.events) == 2
    assert trace.success
```

### E2E Test (with Mock LLM)

**File**: `tests/test_e2e_mock.py`

```python
import pytest
from minime.orchestrator.orchestrator import Orchestrator
from minime.config import MiniMeConfig

@pytest.fixture
def e2e_config(tmp_path):
    """Create config for E2E test."""
    return MiniMeConfig(
        vault_path=str(tmp_path / "vault"),
        db_path=str(tmp_path / "minime.db"),
        default_provider="mock",  # use mock
        trace_dir=str(tmp_path / "logs"),
        config_dir=str(tmp_path / "config"),
    )

@pytest.mark.asyncio
async def test_e2e_task_execution(e2e_config):
    """Full pipeline: task → classification → mask → retrieval → agents → output."""
    
    orchestrator = Orchestrator(e2e_config)
    
    result = await orchestrator.process_task(
        query="Write a simple hello world program",
        provider_override="mock",
        dry_run=True,
    )
    
    assert result.success
    assert len(result.final_output) > 0
    assert len(result.events) > 0  # Check tracing
```

---

## 11) Next 10 Upgrades (Ranked by ROI)

| Rank | Upgrade | ROI | Complexity | Est. Time |
|------|---------|-----|-----------|-----------|
| **1** | **Structured Output Strictness** | High | Low | 1 week |
| **2** | **Better Retrieval Ranking** (BM25 + reranking) | High | Medium | 1 week |
| **3** | **VS Code Integration** (side panel for proposals) | High | Medium | 2 weeks |
| **4** | **Per-Project Profiles** (domain-specific masks + memory scoping) | High | Medium | 1 week |
| **5** | **Evaluation Harness** (automated quality checks) | Medium | Medium | 1 week |
| **6** | **Provider Caching** (LLM response cache by prompt hash) | Medium | Low | 3 days |
| **7** | **Agent Config Packs / Marketplace** (share agent specs) | Medium | Low | 1 week |
| **8** | **Security Hardening** (sandboxed execution + secrets management) | Medium | High | 2 weeks |
| **9** | **Obsidian Graph UX** (proposal review UI, edge explainability) | Medium | High | 2 weeks |
| **10** | **Optional LoRA Adapters** (lightweight model fine-tuning path) | Low | High | 2+ weeks |

### Detailed Plans

#### 1. Structured Output Strictness (Week 1)

**Goal**: Force LLM outputs to match JSON schema; retry on mismatch.

```python
# Implement JSON schema validation + retry
class StructuredOutputValidator:
    async def validate_and_retry(
        self,
        output: str,
        schema: Dict,
        provider,
        max_retries: int = 3,
    ):
        """Validate JSON against schema; retry if invalid."""
        for attempt in range(max_retries):
            try:
                parsed = json.loads(output)
                validate(instance=parsed, schema=schema)  # jsonschema lib
                return parsed
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < max_retries - 1:
                    retry_prompt = f"Previous output was invalid:\n{e}\nFix and respond with valid JSON matching this schema:\n{json.dumps(schema)}"
                    output = await provider.generate(retry_prompt, ...)
                else:
                    raise
        return None
```

#### 2. Better Retrieval Ranking (Week 1–2)

**Goal**: Combine vector similarity + BM25 + recency + graph proximity.

```python
# BM25 ranking over retrieved chunks
from rank_bm25 import BM25Okapi

class HybridRetrieval:
    async def retrieve_hybrid(
        self,
        query: str,
        chunks: List[MemoryChunk],
        alpha: float = 0.7,  # vector weight
    ) -> List[MemoryChunk]:
        """Blend vector similarity + BM25 + graph proximity."""
        
        # Vector scores
        vector_scores = await self._vector_search(query, chunks)
        
        # BM25 scores
        corpus = [c.content for c in chunks]
        bm25 = BM25Okapi([doc.split() for doc in corpus])
        bm25_scores = bm25.get_scores(query.split())
        
        # Combine
        hybrid_scores = [
            alpha * v + (1 - alpha) * b
            for v, b in zip(vector_scores, bm25_scores)
        ]
        
        # Re-rank by graph proximity
        final_scores = await self._apply_graph_reranking(
            chunks,
            hybrid_scores,
        )
        
        return sorted(zip(chunks, final_scores), key=lambda x: x[1], reverse=True)
```

#### 3. VS Code Extension (Week 2–3)

**Goal**: Side panel showing proposals; click to approve.

- Minimal extension (manifest.json + webview)
- Connect to local MiniMe server (REST API)
- Show graph proposals, pending feedback, traces
- One-click approval flow

#### 4. Per-Project Profiles (Week 1–2)

**Goal**: Different identity + memory per project.

```python
class ProjectProfile:
    name: str
    vault_path: str  # project-specific vault
    identity_overrides: Dict[str, float]  # domain-specific weights
    scope_filter: str  # e.g., "project:myproj"
    agent_routing: Dict[str, str]  # per-project agent choices
```

#### 5. Evaluation Harness (Week 1–2)

**Goal**: Automated quality checks on outputs.

```python
class OutputEvaluator:
    async def evaluate(self, output: str, rubric: Dict) -> Dict[str, float]:
        """Score output against rubric (LLM-based)."""
        # Criteria: clarity, completeness, safety, style adherence
        # Use Critic agent to score
        pass
```

#### 6. Provider Caching (1 week)

**Goal**: Cache LLM responses by prompt hash.

```python
class CachedProviderRouter:
    def __init__(self, base_router, cache_dir: str):
        self.base = base_router
        self.cache = {}  # hash → response
    
    async def generate(self, prompt, ...):
        h = hash_prompt(prompt)
        if h in self.cache:
            return self.cache[h]
        result = await self.base.generate(prompt, ...)
        self.cache[h] = result
        return result
```

#### 7. Agent Config Packs (Week 1–2)

**Goal**: Share + reuse agent specs.

```
minime/
  agent-packs/
    biotech-modeling/
      agents.yaml
      prompts/
        architect.txt
        ...
      examples/
        example1.md
```

**Command**: `minime agent install biotech-modeling`

#### 8. Security Hardening (Week 2–3)

**Goal**: Sandboxed execution, secrets management.

- Use `nix-shell` or `podman` for isolated command execution
- Externalize secrets (HuggingFace, API keys) to env vars / `.env`
- No credentials in traces
- Path allowlisting for file operations

#### 9. Obsidian Graph UX (Week 2–3)

**Goal**: Web UI for graph visualization + edge explanation.

```
minime web
  # Launches http://localhost:3000
  # Shows vault as graph
  # Click edges to see rationale
  # Approve/reject proposals
```

**Tech**: React + D3 / Cytoscape

#### 10. Optional LoRA Adapters (Week 3+)

**Goal**: Low-rank fine-tuning path (future, post-MVP).

```python
class LoRAMaskAdapter(nn.Module):
    """Optional: LoRA-style fine-tuning of mask network."""
    # Requires feedback at scale; not in MVP
    pass
```

---

## Summary & Next Steps

### What You Can Deploy Today

```bash
# 1. Clone starter repo
git clone https://github.com/you/minime.git
cd minime

# 2. Install
make install

# 3. Initialize
minime init --mode=dev

# 4. Run first task
minime task "write a hello world" --provider=mock

# 5. View trace
minime trace-view
```

### MVP Deliverables (2–3 weeks)

✅ Identity layer (P_global vectors + config)  
✅ Obsidian vault indexing + SQLite storage  
✅ Graph store + proposal system (auto-approve high-confidence edges, manual approval for low-confidence)  
✅ Mask system (heuristic weights)  
✅ Context Manager (graph-aware retrieval + cold start support)  
✅ Multi-agent orchestration (5 agents)  
✅ Provider routing interface (OpenAI + mock)  
✅ Tool executor (allowlist + risk-based gating: safe auto-execute, high-risk require approval)  
✅ JSONL tracing  
✅ Feedback collection  
✅ CLI (8+ commands)  
✅ E2E tests (mock provider)

### Why This Design Wins

1. **No fine-tuning** → frozen models, fast iteration
2. **Composition over integration** → swap any component
3. **Transparent** → every decision traced + debuggable
4. **Smart gating** → risk-based execution (safe actions auto-execute, high-risk require approval)
5. **Local-first** → privacy + offline capability
6. **Extensible** → graph + masking + agents are pluggable

---

**You're ready to start coding today. The starter code above is copy/paste runnable. Build fearlessly. 🚀**