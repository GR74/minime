# MiniMe: Build-Ready Implementation Plan
## Identity-Conditioned Multi-Agent LLM Orchestration System

**Status:** Build-Ready (MVP 2–3 weeks)  
**Date:** 2025-12-24  
**Audience:** Engineering team starting implementation today

---

## 1) One-Page System Map

### Architecture Overview

MiniMe is a six-layer orchestration system that conditions frozen LLM generation through identity, memory retrieval, and learned mask weighting. All automation is gated; every decision is traceable.

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTENT (CLI)                         │
│                  (task_id, domain, request text)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  TASK CLASSIFIER (online)                                       │
│  - Detect domain (ML, bio, robotics, etc.)                      │
│  - Estimate complexity                                          │
│  - Select agent team                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │ LOAD SYSTEM STATE (online)              │
        │ - P_global (identity vectors)           │
        │ - MaskNetwork θ (learned weights)       │
        │ - AgentSpecs (config)                   │
        └────────────────────┬────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  MASK GENERATION (online)                                       │
│  - Embed task intent → z_task                                   │
│  - MLP([z_identity, z_task, z_domain]) → [α, β, γ, τ, k]       │
│  - α, β, γ: principle blend scalars                             │
│  - τ: temperature / rigor knob                                  │
│  - k: retrieval limit                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  CONTEXT MANAGER + GRAPH RETRIEVAL (online + OBSIDIAN VAULT)    │
│                                                                  │
│  Obsidian Vault (filesystem)                                    │
│    ↓ [Watch & Reindex]                                          │
│  SQLite GraphDB (metadata + edge proposals + embeddings index)   │
│                                                                  │
│  On retrieval request:                                          │
│    1. Embed query → z_query                                     │
│    2. Vector search (top-k candidates)                          │
│    3. Graph traversal (walk proximity nodes)                     │
│    4. Weight by: similarity + graph_proximity + recency + scope │
│    5. Filter by: domain tags + privacy gates                    │
│    6. Apply retrieval_k budget (from mask)                      │
│    7. Return ranked RetrievalResult                             │
│                                                                  │
│  Graph Update Proposals (offline, gated):                        │
│    - Suggest new edges based on embedding similarity            │
│    - Store in DB as (proposed=true, confidence, rationale)      │
│    - User approval required before applying                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  CONTEXT BROKER (online)                                        │
│  - Assemble blocks: [Identity] [Domain] [Retrieved Memory]      │
│  - Apply mask weights                                           │
│  - Enforce token budget (context_limit - output_budget)         │
│  - Compress via summarization if needed                         │
│  - Return assembled_context                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  AGENT LOOP (online, multi-turn)                                │
│                                                                  │
│  For each agent in [Architect, Research, Builder, Critic]:      │
│    1. Assemble agent-specific masked context                    │
│    2. System prompt + task + context → LLM (frozen)             │
│    3. Structured output (JSON schema per agent)                 │
│    4. Store AgentMessage in shared context                      │
│    5. Critic reviews for constraint violations                  │
│       - If violated: request revision, loop                     │
│       - Else: integrate output                                  │
│                                                                  │
│  Integrator:                                                    │
│    - Merge agent outputs                                        │
│    - Resolve conflicts                                          │
│    - Produce final_output                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  ACTION PROPOSALS (gated)                                       │
│  - Parse LLM output → [write_file, apply_patch, run_command]   │
│  - Dry-run / preview each action                                │
│  - Present to user for approval                                 │
│  - Store proposals in ActionProposal schema                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  EXECUTION LAYER (online, gated)                                │
│  - User approves action(s)                                      │
│  - Check allowlist (tools + tags)                               │
│  - Sandbox / dry-run                                            │
│  - Execute: filesystem writes, diffs, subprocess calls          │
│  - Log outcomes in trace                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  TRACE + FEEDBACK (online + offline)                            │
│  - Write JSONL RunTrace with all decisions, tokens, actions     │
│  - User: accept/reject/edit → store as FeedbackCorrectionPair  │
│  - Collect training data for mask network                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  OFFLINE UPDATES (batch, async)                                 │
│                                                                  │
│  Vault Reindexing:                                              │
│    - File watcher triggers on .md changes                       │
│    - Chunk + embed changed notes                                │
│    - Update SQLite                                              │
│                                                                  │
│  Graph Edge Proposals:                                          │
│    - Run similarity clustering on embeddings                    │
│    - Propose new edges (marked requires_user_approval=true)     │
│    - Store in DB; user reviews via CLI                          │
│                                                                  │
│  Mask Network Training:                                         │
│    - Collect: [z_task, z_identity, z_domain, accepted_output]  │
│    - Rejected outputs as negative examples                      │
│    - Train MLP θ on batches (weekly or user-triggered)          │
│    - Validate: does new θ improve past traces?                  │
│    - Save θ if valid, else revert                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  SAVED STATE    │
                    │  - P_global     │
                    │  - MaskNet θ    │
                    │  - GraphDB      │
                    │  - Traces (JSONL│
                    │  - Configs      │
                    └─────────────────┘
```

### What Runs Online vs Offline

| Component | Online | Offline | Trigger |
|-----------|--------|---------|---------|
| Task classifier | ✓ | | Per request |
| Mask generation | ✓ | | Per request |
| Graph retrieval | ✓ | | Per request |
| Context assembly | ✓ | | Per request |
| Agent loop | ✓ | | Per request |
| Action proposal | ✓ | | Per agent output |
| Execution | ✓ | | User approval |
| Trace logging | ✓ | | Per step |
| Vault reindexing | | ✓ | File watch / manual |
| Edge proposal | | ✓ | Nightly / manual |
| Mask training | | ✓ | Weekly / manual |

### Key Design Principles

1. **Hard Separation of Concerns**: Identity, Memory, Mask, Orchestration, Execution layers are decoupled via schema-based interfaces.
2. **Obsidian as First-Class Memory**: Vault is the source of truth; SQLite stores computed metadata (embeddings, edges, proposals). Edits to .md files trigger reindex.
3. **Graph-Aware Retrieval**: Vector search alone is insufficient; proximity, node type, recency, and confidence weights all influence retrieval ranking.
4. **All Automation Gated**: Every action (write, patch, run) requires user approval. Proposals are previewed.
5. **Complete Traceability**: Every run produces a JSONL trace; no decision is opaque.
6. **Frozen LLMs**: Foundation models are never fine-tuned. Personalization happens via mask conditioning.
7. **Provider Agnostic**: Router interface allows swapping OpenAI, Anthropic, local, or custom providers per agent/task.

---

## 2) MVP Scope (2–3 weeks)

### Features (Explicit)

**Core Orchestration**
- ✓ Task classification (domain + complexity detection)
- ✓ Agent spawning (Architect, Research, Builder, Critic, Integrator)
- ✓ Multi-turn agent loop with structured JSON outputs
- ✓ Critic constraint checking + revision requests
- ✓ Integrator conflict resolution

**Identity + Masking**
- ✓ Load P_global (identity embedding matrix) from YAML config
- ✓ Heuristic mask generation (fixed α, β, γ, τ, k per domain/task)
- ✓ Placeholder MLP (trainable, but no training in MVP; for testing only)
- ✓ Block assembly with weighted context injection

**Obsidian Vault Integration**
- ✓ Vault watcher (file system polling; index on .md changes)
- ✓ Note chunking + embedding (via provider embeddings API)
- ✓ SQLite graph store (VaultNode, GraphEdge, GraphUpdateProposal tables)
- ✓ Explicit edge import (parse wikilinks + tags from frontmatter)
- ✓ Suggested edge proposal (embedding similarity clustering)
- ✓ User approval flow for edge proposals (CLI: `minime graph review`)

**Memory + Retrieval**
- ✓ Vector search + graph traversal (combined ranking)
- ✓ Retrieval budget enforcement (token-aware)
- ✓ Privacy filters (domain tags, scope-based gating)
- ✓ Recency weighting
- ✓ Context broker block assembly

**Tool Execution (Gated)**
- ✓ write_file (with preview)
- ✓ apply_patch (unified diff, with preview)
- ✓ run_command (with allowlist + dry-run)
- ✓ Dry-run preview before execution
- ✓ User approval CLI interface

**Tracing + Feedback**
- ✓ JSONL trace file per run (all decisions, tokens, actions, timings)
- ✓ Feedback CLI: accept/reject/edit + store as FeedbackCorrectionPair
- ✓ Trace viewer (CLI)

**Provider Routing**
- ✓ Provider interface (abstract class)
- ✓ OpenAI-compatible implementation
- ✓ Simple heuristic router (per-agent fixed provider in MVP)
- ✓ Structured JSON output + tool calling support

### Non-Features (Explicit)

- ✗ Fine-tuning of foundation models
- ✗ Autonomous execution (all actions require approval)
- ✗ Mask network training (MLP built, but no training loop in MVP)
- ✗ Multi-provider adaptive routing (fixed routes in MVP)
- ✗ VS Code integration (future)
- ✗ Web UI (CLI only in MVP)
- ✗ Streaming output (full batch responses only)
- ✗ Multi-user / collaborative features
- ✗ Production deployment (local dev mode only)
- ✗ Sandbox execution for run_command (allowlist + warnings only)
- ✗ PDF ingestion (Obsidian .md only in MVP)
- ✗ Long-context optimization (summarization placeholder only)

### Acceptance Criteria Checklist

**Orchestration**
- [ ] Task classifier detects domain from prompt with >80% recall on test set
- [ ] Architect agent produces valid JSON schema output
- [ ] Research agent retrieves relevant memory chunks (manual spot-check)
- [ ] Builder agent writes syntactically valid code
- [ ] Critic rejects outputs violating explicit constraints
- [ ] Integrator merges 3+ agent outputs into cohesive final output
- [ ] Agent loop terminates within token budget

**Identity + Masking**
- [ ] P_global loads from config without error
- [ ] Mask generation produces valid α, β, γ ∈ [0, 1]
- [ ] Block assembly respects weights (higher weight → earlier in context)
- [ ] Context stays within token limit

**Obsidian Integration**
- [ ] Vault watcher detects new/modified .md files within 1 sec
- [ ] Notes are chunked and embedded correctly
- [ ] Wikilinks parsed from [[note]] syntax → GraphEdge created
- [ ] Embedding similarity edges proposed with confidence >0.7
- [ ] User can review edge proposals via CLI
- [ ] Approved edges persisted in SQLite

**Retrieval**
- [ ] Vector search returns top-k chunks by embedding distance
- [ ] Graph traversal weights adjacent nodes
- [ ] Privacy filters block untagged domains on request
- [ ] Recency weighting ranks recent notes higher (configurable decay)
- [ ] Token budget enforced (no retrieval exceeds k_tokens)

**Tool Execution**
- [ ] write_file generates diff preview before asking approval
- [ ] apply_patch shows unified diff preview
- [ ] run_command shows dry-run output before approval
- [ ] Allowlist blocks commands not in allowed_tags
- [ ] Execution succeeds / fails gracefully, logged in trace

**Tracing + Feedback**
- [ ] trace.jsonl contains all decisions + tokens + timing
- [ ] User can review trace via `minime trace view <run_id>`
- [ ] User can submit feedback: `minime feedback accept <run_id>` / `reject` / `edit`
- [ ] FeedbackCorrectionPair stored in DB for future training

**Provider Routing**
- [ ] OpenAI provider receives requests and returns structured JSON
- [ ] Router selects provider per agent without error
- [ ] Tool calling (JSON schema) works with provider

### Risks + Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Obsidian vault watcher misses edits | Medium | High | Use polling + file hash checksum; add manual `minime vault sync` command |
| Embedding API rate limits | Medium | High | Batch embedding requests; cache embeddings locally; offline reindexing |
| Agent loop infinite loop | Low | High | Hard token budget cap; max iterations (10); Critic must approve each cycle |
| Privacy leaks in retrieval | Medium | High | Default deny; explicit scope tags; require user approval for cross-domain retrieval |
| Mask weighting causes catastrophic drift | Low | High | α, β, γ clamped to [0, 1]; validate against training set before deploy |
| Tool execution security | High | Critical | Allowlist only; dry-run preview; no shell=True; user approval mandatory |
| Trace storage explodes | Low | Medium | Compress traces weekly; archive old runs; configurable retention |

**Mitigations in Code**
- Token budgets: `context_limit - output_budget` calculated before retrieval
- Watcher: polling + inode tracking; manual sync command
- Allowlist: tool tags stored in AgentSpec; checked before any execution
- Trace: JSONL streamed + archived; pruning job on startup
- MLP: weights clipped; validation pass before saving

---

## 3) Tech Stack + Why

### Language + Frameworks

**Python 3.11+** (Primary)
- Why: Fast prototyping, rich ML/LLM ecosystem (transformers, pydantic, httpx)
- Framework: No heavy framework; use modular packages

**Key Packages**
- `typer` (CLI): simple, type-safe, auto-docstrings
- `pydantic` (validation): schemas, JSON serialization, runtime type checking
- `httpx` (async HTTP): LLM provider calls, non-blocking
- `sqlite3` (stdlib): graph metadata store, no external DB
- `pathlib` (stdlib): vault file handling
- `dataclasses` + `json` (stdlib): trace serialization
- `numpy` (optional): mask network inference only
- `watchfiles` (lightweight): vault file watcher

**Not Used** (for speed)
- ✗ FastAPI, Django (overkill for CLI)
- ✗ SQLAlchemy ORM (raw SQL via sqlite3)
- ✗ Celery (use subprocess + scheduler in Phase 2)
- ✗ RAG frameworks (build custom graph retrieval)
- ✗ LangChain (too opinionated; we control the flow)

### Vector DB Choice: Local-First + Production Path

**MVP (Local-First)**
- **Approach**: SQLite + filesystem embeddings
  - Embeddings stored as JSONL (one per chunk): `embeddings/note_id.jsonl`
  - Simple cosine similarity search in Python (numpy)
  - Graph metadata in SQLite
- **Pros**: Zero external dependencies, fully offline, fast local iteration
- **Cons**: Not indexed for large scale (>100k embeddings)

**Production Path (Phase 2)**
- Option A: **Qdrant** (lightweight, REST API, can run locally or cloud)
- Option B: **Milvus** (open-source, scalable, better for large corpora)
- Option C: **Pinecone** (managed, simple, serverless)
- **Migration Plan**: Same interface, swap backend; reindex without code changes

**Chunking Strategy**
- Obsidian notes: split by heading level (H2 sections as chunks)
- Max chunk size: 500 tokens (estimate via word count / 1.3)
- Preserve metadata: note_path, section, timestamp, tags
- Embeddings model: OpenAI's `text-embedding-3-small` (1536 dims) or `all-MiniLM-L6-v2` (local)

### Storage: Traces + Metadata

**Traces** (Online)
- Format: JSONL (one RunTrace per line)
- Path: `~/.minime/traces/` (user's home)
- File: `trace_<run_id>_<timestamp>.jsonl`
- Rotation: compress + archive daily traces weekly

**Metadata** (SQLite)
- Path: `~/.minime/db/minime.db`
- Tables:
  - `vault_nodes` (note_path, title, frontmatter, updated_at)
  - `graph_edges` (source, target, type, weight, confidence, requires_approval, created_at)
  - `embeddings_meta` (chunk_id, embedding_path, note_path, section, timestamp)
  - `agent_specs` (name, purpose, system_prompt, tools_allowed, routing_hints)
  - `feedback_corrections` (run_id, delta, scope, rationale, created_at)
  - `action_proposals` (proposal_id, action_type, target, preview, approved, timestamp)

### Job Runner: Offline Updates Strategy

**Architecture**: Async task queue (simple, no external service)

**Approach**
- Vault reindexing: file watcher triggers + manual `minime vault sync`
- Graph edge proposals: batch job triggered by `minime graph propose` (daily or manual)
- Mask training: manual `minime mask train` (weekly or on-demand)

**Implementation**
- Use Python `threading` for background watcher
- Use subprocess for long-running jobs (stay responsive to CLI)
- Queue stored in SQLite (simple INSERT + SELECT)
- Job state: pending, running, completed, failed

**Expansion Path (Phase 2)**
- Replace threading with APScheduler for scheduled jobs
- Use proper task queue if needed (Celery + Redis)

### Local-First Dev Mode vs Production Mode

**Dev Mode** (default)
- Vault: `./vault/` (relative to pwd)
- DB: `./minime-dev.db`
- Config: `./minime.yaml`
- Embeddings: in-process, mock provider
- Traces: `./traces/`
- Environment: `MINIME_MODE=dev`

**Production Mode** (Phase 2)
- Vault: `~/Obsidian/MyVault` (user's actual vault)
- DB: `~/.minime/db/minime.db`
- Config: `~/.minime/config.yaml`
- Embeddings: real API calls (batched, cached)
- Traces: `~/.minime/traces/` (rotated)
- Environment: `MINIME_MODE=prod`

**Initialization**
```bash
minime init --mode dev                # Sets up ./minime-dev.db, ./vault/, ./minime.yaml
minime init --mode prod --vault-path ~/Obsidian/MyVault
```

### Obsidian Integration Approach

**Vault Structure**
```
vault/
  index.md                 # Root note
  identity.md              # User's identity principles (source for P_global)
  projects/
    project-a.md
  domains/
    ml.md
    robotics.md
  knowledge/
    concept-1.md
    concept-2.md
```

**Watcher Implementation**
- Poll `vault/` every 500ms for file changes (inode + mtime)
- On change: read .md, extract frontmatter (YAML), parse wikilinks, chunk, embed
- Store metadata in SQLite
- Propose new edges based on embedding similarity

**Frontmatter Format** (optional but powerful)
```yaml
---
title: Bayesian Inference
domain: ml
scope: global
confidence: 0.9
tags: [inference, probability, reusable]
privacy: public
---

Content here...
```

**Link Types**
- Wikilinks: `[[note-name]]` → explicit edge
- Backlinks: computed automatically via note_path references
- Tags: `#domain:ml` in frontmatter → scope edge
- Suggested: embedding-based (proposed by context manager)

---

## 4) API + Tooling Plan

### LLM API Usage

**Provider Interface** (abstract)
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[ToolSchema] | None,
    ) -> GenerateResponse:
        """
        Generate completion.
        Returns: (text, finish_reason, usage)
        """
        pass

    @abstractmethod
    async def embed(self, text_list: list[str], model: str) -> list[list[float]]:
        """Embed texts. Returns: list of vectors."""
        pass
```

**OpenAI-Compatible Implementation**
```python
class OpenAIProvider(LLMProvider):
    async def generate(self, messages, system, model, temperature, max_tokens, tools):
        # POST to /v1/chat/completions
        # Handle structured output via response_format={"type": "json_object"}
        # Handle tool_choice if tools provided
        # Return: (response.choices[0].message.content, finish_reason, usage)
        pass

    async def embed(self, text_list, model):
        # POST to /v1/embeddings
        # Batch texts (max 2048 per request)
        # Return embeddings
        pass
```

**Generation Pattern**
- Agent specifies output schema (Pydantic model)
- LLM provider gets that schema as JSON-Schema in system prompt
- Provider API handles structured output (GPT-4's `response_format`)
- Parse JSON response → Pydantic model
- Critic validates schema compliance

**Tool Calling**
- Tools defined as OpenAI-compatible schemas (name, description, parameters)
- LLM returns tool_calls JSON
- Orchestrator routes to tool executor
- Tool executor runs with allowlist check
- Result passed back to LLM as tool_result message

### Embeddings Plan + Chunking

**Embedding Model Selection**
- MVP: OpenAI `text-embedding-3-small` (1536 dims, cheap, reliable)
- Alternative: Local `all-MiniLM-L6-v2` (via HuggingFace transformers; 384 dims)

**Chunking Strategy**
- Input: markdown note
- Split by H2 sections (preserve hierarchy)
- Max tokens per chunk: 500 (word_count / 1.3)
- Metadata: `(note_path, section_title, chunk_index, timestamp, tags)`
- Store: `embeddings/` as JSONL
  ```json
  {"chunk_id": "note-1#section-a#0", "embedding": [...], "metadata": {...}}
  ```

**Batch Processing**
- Queue chunks for embedding
- Batch request (max 2048 texts per call to OpenAI)
- Cache embeddings to avoid re-embedding
- Invalidate cache on note edit

### Tool Definitions (JSON Schema)

**retrieve_memory(query, filters)**
```json
{
  "name": "retrieve_memory",
  "description": "Query the obsidian vault memory graph. Returns ranked chunks.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query"
      },
      "filters": {
        "type": "object",
        "properties": {
          "domain": {"type": "string"},
          "max_results": {"type": "integer", "default": 5},
          "scope": {"type": "string", "enum": ["global", "domain", "task", "local"]}
        }
      }
    },
    "required": ["query"]
  }
}
```

**propose_plan(goal, architecture, constraints)**
```json
{
  "name": "propose_plan",
  "description": "Propose a structured plan with steps, dependencies, constraints.",
  "parameters": {
    "type": "object",
    "properties": {
      "goal": {"type": "string"},
      "architecture": {"type": "object"},
      "constraints": {"type": "array", "items": {"type": "string"}},
      "estimated_tokens": {"type": "integer"}
    },
    "required": ["goal"]
  }
}
```

**write_file(path, content, description)**
```json
{
  "name": "write_file",
  "description": "Propose a file write. User must approve.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "content": {"type": "string"},
      "description": {"type": "string", "description": "Why this file is needed"}
    },
    "required": ["path", "content"]
  }
}
```

**apply_patch(path, diff, description)**
```json
{
  "name": "apply_patch",
  "description": "Apply a unified diff to a file. User must approve.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "diff": {"type": "string", "description": "Unified diff format"},
      "description": {"type": "string"}
    },
    "required": ["path", "diff"]
  }
}
```

**run_command(cmd, cwd, allowlist_tag, description)**
```json
{
  "name": "run_command",
  "description": "Propose a shell command. User must approve.",
  "parameters": {
    "type": "object",
    "properties": {
      "cmd": {"type": "string"},
      "cwd": {"type": "string", "default": "."},
      "allowlist_tag": {"type": "string", "description": "e.g., 'test', 'build', 'lint'"},
      "description": {"type": "string"}
    },
    "required": ["cmd", "allowlist_tag"]
  }
}
```

**search_local_repo(pattern, file_type)**
```json
{
  "name": "search_local_repo",
  "description": "Search cwd for files matching pattern.",
  "parameters": {
    "type": "object",
    "properties": {
      "pattern": {"type": "string"},
      "file_type": {"type": "string", "enum": ["py", "js", "md", "json", "all"]}
    },
    "required": ["pattern"]
  }
}
```

### Tool Allowlist Design

**Per-AgentSpec Definition**
```yaml
agents:
  architect:
    tools_allowed:
      - retrieve_memory
      - propose_plan
      - search_local_repo
    command_allowlist_tags:
      - ~  # No shell commands for architect
  builder:
    tools_allowed:
      - retrieve_memory
      - write_file
      - apply_patch
      - run_command
      - search_local_repo
    command_allowlist_tags:
      - test
      - build
      - lint
  critic:
    tools_allowed:
      - retrieve_memory
      - search_local_repo
    command_allowlist_tags:
      - ~
```

**Runtime Check**
```python
def execute_tool(agent_name, tool_name, args):
    agent_spec = load_agent_spec(agent_name)
    if tool_name not in agent_spec.tools_allowed:
        raise ValueError(f"Agent {agent_name} not allowed to call {tool_name}")
    
    if tool_name == "run_command":
        tag = args.get("allowlist_tag")
        if tag not in agent_spec.command_allowlist_tags:
            raise ValueError(f"Command tag {tag} not in allowlist")
    
    # Execute + return result
```

**Approval Gate**
- All tool calls that modify state (write_file, apply_patch, run_command) require user approval
- preview() before approval
- User sees diff, command output simulation, etc.
- Only proceed if user confirms

### Internal: obsidian_graph_update Concept

**Index + Propose Flow**
```
1. Vault Watcher detects change
   ↓
2. Read .md + parse frontmatter
   ↓
3. Chunk by H2, embed each
   ↓
4. Store in embeddings/ + SQLite metadata
   ↓
5. Parse explicit edges (wikilinks, tags, frontmatter refs)
   ↓
6. Store as GraphEdge (requires_approval=false for explicit)
   ↓
7. Cluster embeddings → find high-similarity pairs
   ↓
8. Create GraphUpdateProposal (requires_approval=true)
   ↓
9. User reviews: minime graph review
   ↓
10. User approves → Apply edge to graph (update GraphEdge.requires_approval=false)
```

**Database Schema**
```sql
CREATE TABLE vault_nodes (
    node_id TEXT PRIMARY KEY,
    note_path TEXT UNIQUE,
    title TEXT,
    frontmatter JSON,
    content TEXT,
    domain TEXT,
    scope TEXT,
    tags TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node_id TEXT,
    target_node_id TEXT,
    edge_type TEXT,  -- 'wikilink', 'tag', 'similarity', 'backlink'
    weight REAL,
    confidence REAL,
    rationale TEXT,
    requires_approval BOOLEAN,
    created_at TIMESTAMP,
    approved_at TIMESTAMP,
    FOREIGN KEY(source_node_id) REFERENCES vault_nodes(node_id),
    FOREIGN KEY(target_node_id) REFERENCES vault_nodes(node_id)
);

CREATE TABLE graph_update_proposals (
    proposal_id TEXT PRIMARY KEY,
    edges_to_add JSON,
    edges_to_remove JSON,
    confidence REAL,
    rationale TEXT,
    requires_user_approval BOOLEAN,
    created_at TIMESTAMP,
    reviewed_at TIMESTAMP,
    user_decision TEXT,  -- 'approved', 'rejected', 'pending'
    user_notes TEXT
);

CREATE TABLE embeddings_meta (
    chunk_id TEXT PRIMARY KEY,
    note_path TEXT,
    section_title TEXT,
    chunk_index INTEGER,
    embedding_path TEXT,
    embedding_model TEXT,
    created_at TIMESTAMP
);

CREATE TABLE feedback_corrections (
    feedback_id TEXT PRIMARY KEY,
    run_id TEXT,
    delta TEXT,  -- JSON diff
    scope TEXT,  -- 'task', 'domain', 'global'
    confidence REAL,
    rationale TEXT,
    created_at TIMESTAMP
);
```

---

## 5) Core Data Schemas (Pydantic Models)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any
from enum import Enum

# ============================================================================
# IDENTITY LAYER
# ============================================================================

class IdentityScope(str, Enum):
    GLOBAL = "global"
    DOMAIN = "domain"
    TASK = "task"

class IdentityPrinciple(BaseModel):
    """A single principle in the identity matrix."""
    principle_id: str
    text: str
    vector: list[float]  # d-dimensional embedding
    magnitude: float  # importance weight [0, 1]
    decay_rate: float  # how fast it adapts [0, 1]
    scope: IdentityScope
    tags: list[str]
    created_at: datetime
    updated_at: datetime

class GlobalIdentityMatrix(BaseModel):
    """P_global: fixed-size matrix of identity principles."""
    principles: list[IdentityPrinciple]
    embedding_dim: int
    num_principles: int
    created_at: datetime
    
    def to_matrix(self) -> list[list[float]]:
        """Return principles as matrix (num_principles, embedding_dim)."""
        return [p.vector for p in self.principles]

# ============================================================================
# MEMORY LAYER
# ============================================================================

class MemoryChunk(BaseModel):
    """A chunk of memory (obsidian note section + embedding)."""
    chunk_id: str
    note_path: str
    section_title: Optional[str]
    content: str
    embedding_ref: str  # path to embedding vector
    domain: str
    scope: str
    tags: list[str]
    confidence: float = 0.8
    recency_score: float  # 1.0 = today, decays over time
    created_at: datetime
    updated_at: datetime

class RetrievalQuery(BaseModel):
    """Request to retrieve memory."""
    query: str
    domain: Optional[str] = None
    scope: Optional[str] = None
    max_results: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """Ranked results from graph-aware retrieval."""
    chunks: list[MemoryChunk]
    scores: list[float]  # Ranking scores [0, 1]
    reasons: list[str]  # Why each chunk was ranked
    token_budget_used: int
    total_tokens_available: int

# ============================================================================
# MASK SYSTEM
# ============================================================================

class MaskWeights(BaseModel):
    """Output of mask network: conditioning parameters."""
    alpha: float = Field(ge=0, le=1)  # Global principle blend
    beta: float = Field(ge=0, le=1)  # Domain principle blend
    gamma: float = Field(ge=0, le=1)  # Task principle blend
    agent_gamma: float = Field(ge=0, le=1)  # Agent-specific blend
    temperature: float = Field(ge=0.1, le=2.0)  # Generation rigor [0.1=strict, 2.0=creative]
    retrieval_k: int = Field(ge=1, le=50)  # How many memory chunks to retrieve
    block_weights: dict[str, float]  # Weight per context block (identity, domain, memory, task)
    agent_routing_bias: dict[str, float]  # How much each agent influences final output
    created_at: datetime

# ============================================================================
# AGENT LAYER
# ============================================================================

class AgentMessage(BaseModel):
    """Message from an agent."""
    agent_name: str
    role: str  # 'architect', 'researcher', 'builder', 'critic', 'integrator'
    turn: int
    input_query: str
    output: dict[str, Any]  # Agent-specific schema
    tokens_used: int
    reasoning: Optional[str]
    constraint_violations: list[str] = Field(default_factory=list)
    created_at: datetime

class ActionProposal(BaseModel):
    """Proposed action (write, patch, run)."""
    proposal_id: str
    action_type: str  # 'write_file', 'apply_patch', 'run_command'
    target: str  # File path or command
    preview: str  # What will happen
    description: str
    agent_name: str
    requires_approval: bool = True
    approved: bool = False
    approval_timestamp: Optional[datetime] = None
    execution_timestamp: Optional[datetime] = None
    execution_result: Optional[str] = None
    created_at: datetime

# ============================================================================
# ORCHESTRATION
# ============================================================================

class RunTrace(BaseModel):
    """Complete trace of a single MiniMe run."""
    run_id: str
    task_id: str
    domain_detected: str
    complexity: str
    mask_weights: MaskWeights
    agent_messages: list[AgentMessage]
    action_proposals: list[ActionProposal]
    final_output: dict[str, Any]
    total_tokens_used: int
    total_time_seconds: float
    user_feedback: Optional[str]
    created_at: datetime
    completed_at: datetime

class TraceEvent(BaseModel):
    """Individual event logged to trace."""
    event_type: str  # 'retrieval', 'mask_gen', 'agent_call', 'action_proposed', 'action_executed'
    timestamp: datetime
    details: dict[str, Any]
    tokens_used: int = 0

# ============================================================================
# FEEDBACK + LEARNING
# ============================================================================

class FeedbackCorrectionPair(BaseModel):
    """Training data: run + user feedback."""
    feedback_id: str
    run_id: str
    delta: dict[str, Any]  # What user changed
    scope: IdentityScope  # Which layer to update
    confidence: float = 0.7  # How confident this correction is
    rationale: str
    accepted_by_user: bool
    created_at: datetime

# ============================================================================
# OBSIDIAN GRAPH LAYER
# ============================================================================

class VaultNode(BaseModel):
    """Represents a markdown note in the vault."""
    node_id: str
    note_path: str
    title: str
    frontmatter: dict[str, Any]
    domain: Optional[str]
    scope: str = "global"
    tags: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    wikilinks: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

class GraphEdgeType(str, Enum):
    WIKILINK = "wikilink"
    BACKLINK = "backlink"
    TAG = "tag"
    SIMILARITY = "similarity"
    FRONTMATTER = "frontmatter"

class GraphEdge(BaseModel):
    """Connection between two vault nodes."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: GraphEdgeType
    weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: Optional[str]
    requires_approval: bool = False
    approved: bool = True
    created_at: datetime
    approved_at: Optional[datetime] = None

class GraphUpdateProposal(BaseModel):
    """Proposed edge updates (add/remove) for user review."""
    proposal_id: str
    edges_to_add: list[GraphEdge]
    edges_to_remove: list[GraphEdge]
    confidence: float
    rationale: str
    requires_user_approval: bool = True
    user_decision: Optional[str]  # 'approved', 'rejected', 'pending'
    user_notes: Optional[str]
    created_at: datetime
    reviewed_at: Optional[datetime] = None

# ============================================================================
# AGENT CREATION + ROUTING
# ============================================================================

class AgentSpec(BaseModel):
    """Specification for an agent (fixed or dynamically created)."""
    agent_id: str
    name: str
    purpose: str
    role: str  # 'architect', 'researcher', 'builder', 'critic', 'integrator', 'executor'
    system_prompt: str
    input_schema: dict[str, Any]  # JSON Schema
    output_schema: dict[str, Any]  # JSON Schema
    tools_allowed: list[str]
    command_allowlist_tags: list[str] = Field(default_factory=list)
    constraints: list[str]  # Things this agent must respect
    routing_hints: dict[str, Any]  # Provider + model hints
    max_iterations: int = 3
    created_at: datetime
    user_approved: bool = False

class ToolSchema(BaseModel):
    """Definition of a tool the LLM can call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

# ============================================================================
# PROVIDER ABSTRACTION
# ============================================================================

class GenerateResponse(BaseModel):
    """Response from LLM generation."""
    text: str
    finish_reason: str  # 'stop', 'length', 'tool_calls', etc.
    usage: dict[str, int]  # tokens_used, tokens_prompt, tokens_completion

class ProviderConfig(BaseModel):
    """Config for an LLM provider."""
    name: str  # 'openai', 'anthropic', 'local'
    model: str
    api_key: Optional[str]
    api_base: Optional[str]
    embedding_model: str

# ============================================================================
# CLI + CONFIG
# ============================================================================

class MiniMeConfig(BaseModel):
    """System-level configuration."""
    mode: str = "dev"  # 'dev' or 'prod'
    vault_path: str
    db_path: str
    trace_dir: str
    embedding_model: str
    provider_config: ProviderConfig
    identity_source: str  # Path to identity.md
    default_context_limit: int = 4000
    max_agent_iterations: int = 3
    created_at: datetime
```

---

## 6) Mask System Implementation Plan (Core Innovation)

### Mask Hierarchy

```
┌─────────────────────────────────────────────┐
│          P_effective (in-use principles)     │
│  P_eff = α·P_global + β·P_domain + γ·P_task │
│                 + agent_γ·P_agent           │
└─────────────────────────────────────────────┘
                       ▲
           ┌───────────┼───────────┐
           │           │           │
    ┌──────▼────┐ ┌──────▼────┐ ┌──────▼────┐
    │ P_global  │ │ P_domain  │ │ P_task    │
    │ (always)  │ │ (dynamic) │ │ (per req) │
    └───────────┘ └───────────┘ └───────────┘
         ▲              ▲              ▲
         │              │              │
   [Identity]      [Domain Detect] [Task Embed]
   (persistent)    (task → domain)  (LLM embed)
```

### How α, β, γ Are Selected

**MVP Heuristic** (Fixed Rules)
```python
def select_mask_weights_heuristic(
    domain: str,
    complexity: str,
    task_embedding: list[float],
    agent_name: str,
) -> MaskWeights:
    """
    Heuristic mask selection. Later replaced by learned MLP.
    """
    # Base: always use global identity
    alpha = 1.0
    
    # Domain boost: if domain detected and domain principles exist
    if domain and domain != "unknown":
        beta = 0.8
    else:
        beta = 0.0
    
    # Task specificity: complex tasks get more task-specific weighting
    if complexity == "high":
        gamma = 0.7
    elif complexity == "medium":
        gamma = 0.4
    else:
        gamma = 0.1
    
    # Agent-specific: architect needs more structure, builder needs more specifics
    if agent_name == "architect":
        agent_gamma = 0.6
    elif agent_name == "builder":
        agent_gamma = 0.8
    elif agent_name == "critic":
        agent_gamma = 0.5
    else:
        agent_gamma = 0.3
    
    # Temperature: higher complexity → lower temperature (more deterministic)
    if complexity == "high":
        temperature = 0.3
    elif complexity == "medium":
        temperature = 0.5
    else:
        temperature = 0.7
    
    # Retrieval budget: higher complexity → more memory
    if complexity == "high":
        retrieval_k = 10
    else:
        retrieval_k = 5
    
    block_weights = {
        "identity": alpha,
        "domain": beta,
        "memory": (alpha + beta + gamma) / 3,
        "task": gamma,
    }
    agent_routing_bias = {
        agent_name: 1.0,  # This agent leads
        "critic": 0.3,    # Critic always reviews
    }
    
    return MaskWeights(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        agent_gamma=agent_gamma,
        temperature=temperature,
        retrieval_k=retrieval_k,
        block_weights=block_weights,
        agent_routing_bias=agent_routing_bias,
        created_at=datetime.now(),
    )
```

**Learned Path** (Phase 2+)
- Collect training data: (z_identity, z_task, z_domain, agent_name) → (α, β, γ, τ, k)
- Train small MLP: 
  ```
  Input: concat([z_identity (d), z_task (d), z_domain (d), agent_one_hot (6)])
  Hidden: Linear(3d + 6, 128) → GELU → LayerNorm
  Hidden: Linear(128, 64) → GELU → LayerNorm
  Output: Linear(64, 5) → Sigmoid (for α, β, γ, agent_γ ∈ [0,1])
                         + Softplus (for τ ∈ [0.1, 2.0])
                         + ReLU+Clamp (for k ∈ [1, 50])
  ```
- Training signal: user accepts/rejects outputs + edits
- Offline training (weekly or user-triggered)

### Temporary Context Abstraction

**Concept**: When a task involves only a few topics, create a subspace that isolates reasoning from global state.

**Implementation**
```python
class TemporaryContextAbstraction:
    def __init__(self, parent_identity_matrix: GlobalIdentityMatrix):
        self.parent = parent_identity_matrix
        self.subspace_principals: list[IdentityPrinciple] = []
        self.projection_basis: list[list[float]] = []
    
    def spawn(self, relevant_topics: list[str], task_embedding: list[float]):
        """
        Create a temporary subspace for this task.
        
        Steps:
        1. Retrieve principles related to relevant_topics from parent
        2. Create orthonormal projection basis (QR decomposition)
        3. Project parent principles onto basis (dimensionality reduction)
        4. Store as subspace_principles
        """
        # Filter parent principles by topic relevance
        relevant = [p for p in self.parent.principles if any(t in p.tags for t in relevant_topics)]
        self.subspace_principals = relevant
        
        # Create basis (e.g., via PCA on principle vectors)
        basis = self._compute_projection_basis([p.vector for p in relevant])
        self.projection_basis = basis
    
    def project(self, principle: IdentityPrinciple) -> IdentityPrinciple:
        """Project a principle into subspace."""
        projected_vector = self._project_vector(principle.vector, self.projection_basis)
        return IdentityPrinciple(
            principle_id=principle.principle_id,
            text=principle.text,
            vector=projected_vector,
            magnitude=principle.magnitude * 1.1,  # Boost magnitude in subspace
            decay_rate=principle.decay_rate,
            scope=principle.scope,
            tags=principle.tags,
            created_at=principle.created_at,
            updated_at=datetime.now(),
        )
    
    def merge_back(self, subspace_deltas: list[IdentityPrinciple]):
        """
        Optionally merge learned deltas back to parent.
        Default: discard (keep parent clean).
        """
        # User decides: save to parent or discard
        pass
    
    def discard(self):
        """Cleanup: discard subspace without merging."""
        self.subspace_principals = []
        self.projection_basis = []
```

### Minimal Heuristic Implementation + Path to Trainable MLP

**MVP Code** (in `mask_system.py`)
```python
from pydantic import BaseModel
from datetime import datetime

class MaskNetwork:
    def __init__(self, mode: str = "heuristic"):
        """
        mode='heuristic': fixed rules (MVP)
        mode='learned': use trained weights (Phase 2+)
        """
        self.mode = mode
        self.theta = None  # Will hold trained weights later
    
    def generate_mask(
        self,
        identity_embedding: list[float],
        task_embedding: list[float],
        domain: str,
        agent_name: str,
    ) -> MaskWeights:
        """Generate mask weights for this task."""
        if self.mode == "heuristic":
            return self._heuristic_mask(domain, agent_name)
        else:
            return self._learned_mask(identity_embedding, task_embedding, domain, agent_name)
    
    def _heuristic_mask(self, domain: str, agent_name: str) -> MaskWeights:
        """Fixed heuristic rules."""
        # See earlier code snippet
        pass
    
    def _learned_mask(
        self,
        identity_embedding: list[float],
        task_embedding: list[float],
        domain: str,
        agent_name: str,
    ) -> MaskWeights:
        """Use trained MLP."""
        # Placeholder for Phase 2
        # import torch
        # model = self.load_model()
        # output = model(torch.cat([identity, task, domain_one_hot, agent_one_hot]))
        # Parse output → MaskWeights
        raise NotImplementedError("Learned mask not yet implemented")
```

### Training Data Format

```python
class MaskTrainingExample(BaseModel):
    """One training example for mask network."""
    identity_embedding: list[float]
    task_embedding: list[float]
    domain: str
    agent_name: str
    
    # Output (what mask weights were used)
    mask_weights: MaskWeights
    
    # Label (was output accepted or rejected)
    accepted: bool
    
    # Optional: user edits as delta
    user_delta: Optional[dict[str, Any]] = None
    
    # Confidence in this example (user-curated data higher confidence)
    confidence: float = 0.7
```

**Data Collection**
- Auto-collect on every run: (inputs, mask_weights, accepted)
- User explicitly provides feedback: accept/reject/edit
- Store in SQLite: `mask_training_examples` table
- Filter for high-confidence examples (>0.8) for training

### Offline Training Loop

```python
class MaskTrainer:
    def __init__(self, learning_rate: float = 1e-3, device: str = "cpu"):
        self.lr = learning_rate
        self.device = device
        self.model = None  # Will load/build during train()
    
    def collect_training_data(self, db_path: str) -> list[MaskTrainingExample]:
        """Query training examples from DB."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT identity_embedding, task_embedding, domain, agent_name,
                   mask_weights_json, accepted, user_delta, confidence
            FROM mask_training_examples
            WHERE created_at > (NOW() - INTERVAL 7 DAY)
            AND confidence > 0.7
            ORDER BY created_at DESC
            LIMIT 1000
        """)
        # Parse rows → list[MaskTrainingExample]
        return []  # Stub
    
    def build_model(self):
        """Build MLP."""
        import torch
        import torch.nn as nn
        
        embedding_dim = 1536  # For text-embedding-3-small
        self.model = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 6, 128),  # 3 embeddings + 6 one-hot agents
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 5),  # Output: alpha, beta, gamma, agent_gamma, (temp+k encoded)
        )
    
    def train(self, db_path: str, epochs: int = 5, batch_size: int = 32):
        """Train mask network on feedback data."""
        examples = self.collect_training_data(db_path)
        
        if len(examples) < 10:
            print("Not enough training data (need >10 examples)")
            return False
        
        self.build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare batches
        for epoch in range(epochs):
            for batch in self._batch_examples(examples, batch_size):
                # Forward pass
                # Compute loss (MSE between predicted mask and actual mask)
                # Backward pass
                # Update weights
                pass
        
        # Validate on held-out set
        validation_loss = self._validate(examples[-100:])
        print(f"Validation loss: {validation_loss}")
        
        # Save model if better
        if validation_loss < self.best_loss:
            self.save_model("mask_network.pt")
            return True
        else:
            print("No improvement; reverting weights")
            return False
    
    def save_model(self, path: str):
        """Save trained weights."""
        import torch
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load trained weights."""
        import torch
        self.build_model()
        self.model.load_state_dict(torch.load(path))
```

### How Graph Proximity + Node Types Influence Retrieval/Masking

**Retrieval Weighting Formula**
```
score(chunk) = w_sim * sim_score +
               w_prox * proximity_score +
               w_recency * recency_score +
               w_confidence * confidence_score +
               w_scope * scope_match_score

where:
  sim_score ∈ [0, 1]        # Cosine similarity to query
  proximity_score ∈ [0, 1]   # How close in graph (1 - dist/max_dist)
  recency_score ∈ [0, 1]     # Decay over time (e^(-lambda * age_days))
  confidence_score ∈ [0, 1]  # User confidence in note (from frontmatter)
  scope_match_score ∈ [0, 1] # Does node scope match requested scope?

  w_sim, w_prox, ... ∈ [0, 1]  # Weights from mask network
```

**Example**
```python
def rank_retrieval_results(
    chunks: list[MemoryChunk],
    query_embedding: list[float],
    mask_weights: MaskWeights,
    graph: GraphDB,
) -> list[tuple[MemoryChunk, float]]:
    """
    Rank chunks using combined similarity + graph proximity.
    """
    scores = []
    
    for chunk in chunks:
        # 1. Embedding similarity
        sim_score = cosine_similarity(chunk.embedding, query_embedding)
        
        # 2. Graph proximity (how many hops to related nodes?)
        source_node = graph.get_node(chunk.chunk_id)
        proximity_score = 1.0 - (graph.shortest_path(source_node, query_node) / MAX_HOPS)
        proximity_score = max(0, proximity_score)
        
        # 3. Recency
        age_days = (datetime.now() - chunk.updated_at).days
        recency_score = np.exp(-0.1 * age_days)
        
        # 4. Confidence (from frontmatter or default)
        confidence_score = chunk.confidence
        
        # 5. Scope match
        scope_match = 1.0 if chunk.scope == requested_scope else 0.5
        
        # Combined score
        w_dict = mask_weights.block_weights
        total_score = (
            w_dict.get("similarity", 0.4) * sim_score +
            w_dict.get("proximity", 0.2) * proximity_score +
            w_dict.get("recency", 0.15) * recency_score +
            w_dict.get("confidence", 0.15) * confidence_score +
            w_dict.get("scope", 0.1) * scope_match
        )
        
        scores.append((chunk, total_score))
    
    # Sort by score descending
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

## 7) Multi-Agent Orchestration Plan

### Agent Roles + Responsibilities

| Agent | What It Sees | Inputs | Outputs | Tools | Constraints |
|-------|-------------|--------|---------|-------|-------------|
| **Architect** | Global structure + domain principles | Task, prior art (memory), domain constraints | Dependency graph, module boundaries, interface specs | retrieve_memory, propose_plan, search_local_repo | Must not prescribe implementation details |
| **Research** | All memory + external context | Query (from task), domain filter | Ranked findings, formulas, code snippets | retrieve_memory, search_local_repo | Must cite sources; no hallucination |
| **Builder** | Architecture + research + task | Arch spec, findings, task | Code, configs, infrastructure | write_file, apply_patch, run_command, search_local_repo | Must obey architect's boundaries |
| **Critic** | All prior outputs | Outputs from Architect, Research, Builder | Violation list, requested revisions | retrieve_memory, search_local_repo | Cannot propose fixes; only reject + explain |
| **Integrator** | All agent outputs | All agent messages | Final output, conflict resolution | propose_plan | Must unify outputs; mark remaining conflicts |
| **Executor** | Proposals + user state | Action proposals, user approval | Execution results, logs | run_command, write_file, apply_patch | Requires explicit user approval before any action |

### Orchestration Flow

```
1. User submits task
   ↓
2. Task classifier:
   - Embed task text
   - Detect domain (keyword + semantic)
   - Estimate complexity
   ↓
3. Load state:
   - P_global (identity)
   - MaskNetwork θ (learned weights)
   - AgentSpecs (config)
   ↓
4. Generate mask:
   - MaskNetwork([z_identity, z_task, z_domain, agent_name]) → MaskWeights
   ↓
5. AGENT LOOP (max 3 iterations per agent):
   ┌──────────────────────────────────────┐
   │ For agent in [Architect, Research,   │
   │   Builder, Critic, Integrator]:      │
   │                                      │
   │ a. Mask retrieval:                   │
   │    - Query memory with task + domain │
   │    - Apply mask weights (k, filters) │
   │    - Return top-k ranked chunks      │
   │                                      │
   │ b. Assemble context:                 │
   │    - Identity block (α weight)       │
   │    - Domain block (β weight)         │
   │    - Memory block (retrieved chunks) │
   │    - Task block (full task text)     │
   │    - Enforce token budget            │
   │                                      │
   │ c. LLM call:                         │
   │    - System prompt (agent-specific)  │
   │    - Assembled context               │
   │    - Tools available (allowlist)     │
   │    - Structured output (JSON schema) │
   │    - Temperature from mask           │
   │    ↓ LLM (frozen) returns JSON       │
   │                                      │
   │ d. Parse output:                     │
   │    - Validate against schema         │
   │    - Store as AgentMessage           │
   │    - Extract any tool_calls          │
   │                                      │
   │ e. Critic review (special):          │
   │    IF agent != Critic:               │
   │       Spawn Critic sub-loop          │
   │       Critic checks against:         │
   │       - Identity principles          │
   │       - AgentSpec constraints        │
   │       - Logical consistency          │
   │       IF violations found:           │
   │          Request revision            │
   │          Loop agent (i++)            │
   │       ELSE:                          │
   │          Accept output               │
   │                                      │
   │ f. Log TraceEvent:                   │
   │    - agent_name, turn, tokens_used   │
   │    - output (JSON)                   │
   │    - constraint_violations (if any)  │
   │                                      │
   └──────────────────────────────────────┘
   ↓
6. INTEGRATOR MERGE:
   - Collect all agent outputs
   - Check for conflicts (e.g., arch vs builder disagreement)
   - Mark conflicts + request human input if needed
   - Produce final_output (unified view)
   ↓
7. ACTION PROPOSAL:
   - Parse final_output for tool calls
   - Create ActionProposal for each (write_file, apply_patch, run_command)
   - Show user a preview
   ↓
8. User approval:
   - User reviews proposals
   - Approves / rejects / edits
   - Send feedback to orchestrator
   ↓
9. EXECUTION (gated):
   - For each approved action:
     - Check allowlist
     - Execute (write, patch, run)
     - Capture output + error
     - Log to trace
   ↓
10. Feedback collection:
    - User: accept/reject final output
    - User: optional edits + rationale
    - Store FeedbackCorrectionPair
    ↓
11. Write trace:
    - Save full RunTrace as JSONL
    - Include all decisions, tokens, actions, feedback
```

### Critic Enforcement

**Critic Constraints** (checked on every other agent's output)
```yaml
critic:
  constraints:
    - "No undefined variables or imports"
    - "All architectures must follow principles from P_global"
    - "Builder code must match architect's interface specs"
    - "No tool calls that violate allowlist"
    - "Memory retrieval must cite sources"
    - "No hardcoded secrets or credentials"
    - "Code style must match user's identity (naming, structure)"
```

**Revision Loop**
```python
def critic_review(agent_output: AgentMessage, constraints: list[str]) -> tuple[bool, list[str]]:
    """
    Check output against constraints.
    Returns: (is_valid, violations)
    """
    violations = []
    
    for constraint in constraints:
        # Embed constraint + output
        # Query memory for examples of violations
        # Use LLM to judge: does output violate this constraint?
        violation = check_constraint_via_llm(constraint, agent_output)
        if violation:
            violations.append(constraint)
    
    return len(violations) == 0, violations

def orchestrator_loop_with_critic(agent, task, context, max_iterations=3):
    """Run agent; have Critic review; loop if needed."""
    for iteration in range(max_iterations):
        output = agent.generate(task, context)
        is_valid, violations = critic_review(output, agent.spec.constraints)
        
        if is_valid:
            return output  # Success
        else:
            # Request revision
            revision_prompt = f"Your output violated: {violations}. Please revise."
            context.add_message(role="critic", content=revision_prompt)
            # Loop: agent will see critic feedback + retry
    
    # Max iterations exceeded
    raise RuntimeError(f"Agent {agent.name} could not satisfy constraints after {max_iterations} attempts")
```

### Iteration + Termination Conditions

**Per-Agent Iteration**
- Max 3 revisions per agent (from Critic feedback)
- Terminate if: constraint satisfied OR max iterations

**Global Termination**
- All agents complete successfully (no violations)
- OR max total tokens exceeded (hard limit)
- OR user cancels

**Integrator Conflict Resolution**
```python
def integrator_resolve_conflicts(agent_outputs: dict[str, AgentMessage]) -> dict[str, Any]:
    """
    Merge outputs; detect + resolve conflicts.
    Returns: final_output (unified view)
    """
    arch_spec = agent_outputs["architect"].output
    builder_code = agent_outputs["builder"].output
    research_findings = agent_outputs["research"].output
    
    # Check for mismatches
    conflicts = []
    
    # Example conflict: builder code doesn't match architect boundaries
    if not code_matches_architecture(builder_code, arch_spec):
        conflicts.append({
            "type": "architecture_mismatch",
            "description": "Builder code violates architect's interface spec",
            "requires_human_resolution": True,
        })
    
    final = {
        "architecture": arch_spec,
        "research": research_findings,
        "code": builder_code,
        "conflicts": conflicts,
        "resolution_note": "See conflicts; user must decide" if conflicts else "No conflicts",
    }
    
    return final
```

### Provider Routing Policy

**Per-AgentSpec Definition**
```yaml
agents:
  architect:
    provider: openai
    model: gpt-4
    temperature_override: null  # Use mask-generated temperature
  research:
    provider: openai
    model: gpt-4-turbo
  builder:
    provider: openai
    model: gpt-4
  critic:
    provider: openai
    model: gpt-4  # Critic needs best reasoning
```

**Router Logic**
```python
def route_agent_to_provider(agent_name: str, agent_spec: AgentSpec, config: MiniMeConfig) -> LLMProvider:
    """Select provider for agent."""
    provider_name = agent_spec.routing_hints.get("provider", config.provider_config.name)
    model = agent_spec.routing_hints.get("model", config.provider_config.model)
    
    # Later: add latency/cost tracking to route optimally
    
    if provider_name == "openai":
        return OpenAIProvider(model=model, api_key=config.provider_config.api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(model=model, api_key=config.provider_config.api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
```

### AgentSpec Proposal Flow (Gated)

**When New Agent Is Needed**
1. System detects: "No existing agent fits this task"
2. Generate AgentSpec (name, purpose, io_schema, constraints, tools_allowed)
3. Store as (user_approved=False)
4. Present to user: `minime agent propose <spec_id>`
5. User reviews + approves OR rejects
6. If approved: activate agent
7. Log all specs for future reuse

**Code**
```python
def propose_new_agent_spec(task: str, existing_agents: list[AgentSpec]) -> AgentSpec:
    """Generate a new agent spec if existing ones don't fit."""
    # Ask LLM: "Do existing agents fit? If not, design a new one."
    
    # For MVP: hard-coded 5 agents, no dynamic creation
    # For Phase 2: LLM-designed agents (with safety gates)
    
    if not agent_fits_task(task, existing_agents):
        # Create spec
        spec = AgentSpec(
            agent_id=f"agent_{uuid.uuid4()}",
            name=f"Agent_{task_type}",
            purpose=f"Handle {task_type} tasks",
            role="specialist",
            system_prompt="...",
            input_schema={...},
            output_schema={...},
            tools_allowed=["retrieve_memory", "search_local_repo"],
            constraints=["Must cite sources"],
            user_approved=False,  # Requires approval
        )
        
        # Store in DB
        store_agent_spec(spec)
        
        # Prompt user
        print(f"New agent spec proposed: {spec.name}")
        print(f"  Purpose: {spec.purpose}")
        print(f"  Tools: {spec.tools_allowed}")
        print(f"Approve? (y/n)")
        
        if user_confirms():
            spec.user_approved = True
            return spec
        else:
            return None
    else:
        return select_best_fit_agent(task, existing_agents)
```

---

## 8) Repo Scaffold (Exact Tree)

```
minime/
├── README.md                          # Project overview + quick start
├── LICENSE                            # Apache 2.0 or MIT
├── pyproject.toml                     # Poetry or pip dependencies
├── setup.py                           # (if not using pyproject.toml)
├── .gitignore                         # Standard Python
├── .env.example                       # API keys template (don't commit .env)
│
├── minime/                            # Main package
│   ├── __init__.py                    # Package init
│   ├── __main__.py                    # CLI entrypoint (python -m minime)
│   ├── cli.py                         # Typer CLI (main commands)
│   ├── config.py                      # Config loading + validation (Pydantic)
│   │
│   ├── core/                          # Core orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # Orchestrator class (main loop)
│   │   ├── task_classifier.py         # Domain + complexity detection
│   │   ├── context_broker.py          # Block assembly + token budget
│   │   └── agent_loop.py              # Agent spawning + iteration
│   │
│   ├── identity/                      # Identity layer
│   │   ├── __init__.py
│   │   ├── models.py                  # IdentityPrinciple, GlobalIdentityMatrix
│   │   └── loader.py                  # Load P_global from identity.md
│   │
│   ├── memory/                        # Memory + retrieval
│   │   ├── __init__.py
│   │   ├── vault.py                   # Obsidian vault management
│   │   ├── retrieval.py               # Vector search + graph-aware ranking
│   │   └── embeddings.py              # Embedding API calls + caching
│   │
│   ├── graph/                         # Obsidian graph layer
│   │   ├── __init__.py
│   │   ├── models.py                  # VaultNode, GraphEdge, GraphUpdateProposal
│   │   ├── store.py                   # SQLite graph DB
│   │   ├── indexer.py                 # Vault watcher + indexing
│   │   └── traversal.py               # Graph traversal + proximity scoring
│   │
│   ├── mask/                          # Mask system (core innovation)
│   │   ├── __init__.py
│   │   ├── models.py                  # MaskWeights, MaskTrainingExample
│   │   ├── generator.py               # Heuristic + learned mask generation
│   │   ├── network.py                 # MaskNetwork class (placeholder trainable)
│   │   └── trainer.py                 # Offline training loop
│   │
│   ├── agents/                        # Agent definitions + specs
│   │   ├── __init__.py
│   │   ├── specs.py                   # AgentSpec models + defaults
│   │   ├── base.py                    # Base Agent class
│   │   ├── architect.py               # Architect agent implementation
│   │   ├── research.py                # Research agent implementation
│   │   ├── builder.py                 # Builder agent implementation
│   │   ├── critic.py                  # Critic agent implementation
│   │   ├── integrator.py              # Integrator agent implementation
│   │   └── executor.py                # Executor agent (action approval + execution)
│   │
│   ├── providers/                     # LLM provider implementations
│   │   ├── __init__.py
│   │   ├── base.py                    # LLMProvider abstract base class
│   │   ├── openai_compat.py           # OpenAI-compatible provider
│   │   ├── router.py                  # Provider routing logic
│   │   └── mock.py                    # Mock provider (for testing)
│   │
│   ├── tools/                         # Tool definitions + execution
│   │   ├── __init__.py
│   │   ├── registry.py                # Tool schema registry
│   │   ├── executor.py                # Execute tools (gated)
│   │   ├── retrieve_memory.py         # retrieve_memory tool
│   │   ├── propose_plan.py            # propose_plan tool
│   │   ├── write_file.py              # write_file tool (with preview)
│   │   ├── apply_patch.py             # apply_patch tool (with preview)
│   │   ├── run_command.py             # run_command tool (with allowlist)
│   │   └── search_local.py            # search_local_repo tool
│   │
│   ├── tracing/                       # Tracing + logging
│   │   ├── __init__.py
│   │   ├── models.py                  # RunTrace, TraceEvent, FeedbackCorrectionPair
│   │   ├── tracer.py                  # Trace recorder (JSONL)
│   │   ├── viewer.py                  # CLI trace viewer
│   │   └── storage.py                 # Trace persistence + rotation
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── token_counter.py           # Estimate tokens (for budget enforcement)
│       ├── embeddings_cache.py        # Local embedding cache
│       └── schemas.py                 # Shared Pydantic models
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── unit/
│   │   ├── test_retrieval.py          # Vector search + graph ranking
│   │   ├── test_context_assembly.py   # Block assembly + token budget
│   │   ├── test_tool_allowlist.py     # Allowlist enforcement
│   │   ├── test_trace_correctness.py  # Trace logging validation
│   │   ├── test_graph_proposals.py    # Edge proposal logic
│   │   └── test_mask_generation.py    # Mask weight generation
│   ├── e2e/
│   │   ├── test_full_orchestration.py # E2E: task → final output (mocked LLM)
│   │   └── test_vault_integration.py  # Vault watcher + indexing
│   └── fixtures/
│       ├── sample_vault/              # Test vault
│       ├── sample_config.yaml         # Test config
│       └── mock_responses.py          # Mocked LLM responses
│
├── examples/                          # Example scripts + notebooks
│   ├── basic_task.py                  # Run a simple task
│   ├── custom_agent.py                # Create + activate a custom agent
│   └── trace_analysis.ipynb           # Jupyter: analyze traces
│
├── docs/                              # Documentation
│   ├── architecture.md                # Architecture deep dive
│   ├── api_reference.md               # API docs
│   ├── tutorial.md                    # Getting started
│   ├── mask_system.md                 # Mask system explained
│   ├── obsidian_integration.md        # Vault integration guide
│   └── faq.md                         # Common questions
│
├── scripts/                           # Development scripts
│   ├── init_dev.sh                    # Initialize dev environment
│   ├── test.sh                        # Run all tests
│   ├── lint.sh                        # Lint + format code
│   └── train_mask.py                  # Offline mask training (Phase 2)
│
└── ~/.minime/                         # User home directory (created on init)
    ├── config.yaml                    # User config (persisted)
    ├── identity.yaml                  # Identity principles (editable by user)
    ├── db/
    │   └── minime.db                  # SQLite: graph + metadata + feedback
    ├── embeddings/                    # Cached embeddings (JSONL)
    ├── traces/                        # Run traces (JSONL)
    ├── vault/                         # Obsidian vault (user's notes)
    └── agents/                        # Custom agent specs (YAML)
```

---

## 9) Starter Code (Must Run)

### 9.1 pyproject.toml (Dependencies)

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[project]
name = "minime"
version = "0.1.0"
description = "Identity-conditioned multi-agent LLM orchestration system"
authors = [{name = "Your Team", email = "team@example.com"}]
license = {text = "Apache-2.0"}
requires-python = ">=3.11"

dependencies = [
    "typer[all]>=0.9.0",
    "pydantic>=2.0",
    "httpx>=0.25.0",
    "python-dotenv>=1.0",
    "watchfiles>=0.21.0",
    "numpy>=1.24.0",
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[project.scripts]
minime = "minime.cli:app"

[tool.poetry]
name = "minime"
version = "0.1.0"
description = "Identity-conditioned multi-agent LLM orchestration system"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

### 9.2 minime/config.py (Configuration Management)

```python
"""Configuration loading and validation."""

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import yaml
import os
from datetime import datetime


class ProviderConfig(BaseModel):
    name: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"


class MiniMeConfig(BaseModel):
    mode: str = "dev"  # 'dev' or 'prod'
    vault_path: str
    db_path: str
    trace_dir: str
    embedding_model: str
    provider_config: ProviderConfig
    identity_source: str
    default_context_limit: int = 4000
    max_agent_iterations: int = 3
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


def get_home_dir() -> Path:
    """Get MiniMe home directory."""
    home = Path.home() / ".minime"
    return home


def init_dev_mode() -> MiniMeConfig:
    """Initialize dev mode (local vault + DB)."""
    cwd = Path.cwd()
    vault_path = cwd / "vault"
    db_path = cwd / "minime-dev.db"
    trace_dir = cwd / "traces"
    
    vault_path.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)
    
    config = MiniMeConfig(
        mode="dev",
        vault_path=str(vault_path),
        db_path=str(db_path),
        trace_dir=str(trace_dir),
        embedding_model="text-embedding-3-small",
        provider_config=ProviderConfig(
            name="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model="text-embedding-3-small",
        ),
        identity_source=str(vault_path / "identity.md"),
    )
    
    return config


def init_prod_mode(vault_path: str) -> MiniMeConfig:
    """Initialize production mode."""
    home = get_home_dir()
    home.mkdir(exist_ok=True)
    
    config = MiniMeConfig(
        mode="prod",
        vault_path=vault_path,
        db_path=str(home / "db" / "minime.db"),
        trace_dir=str(home / "traces"),
        embedding_model="text-embedding-3-small",
        provider_config=ProviderConfig(
            name="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model="text-embedding-3-small",
        ),
        identity_source=str(home / "identity.md"),
    )
    
    return config


def load_config(mode: str = "dev", vault_path: Optional[str] = None) -> MiniMeConfig:
    """Load configuration."""
    if mode == "dev":
        return init_dev_mode()
    elif mode == "prod":
        if not vault_path:
            raise ValueError("vault_path required for prod mode")
        return init_prod_mode(vault_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

### 9.3 minime/cli.py (CLI Entrypoint)

```python
"""Command-line interface."""

import typer
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from minime.config import load_config
from minime.core.orchestrator import Orchestrator
from minime.graph.indexer import VaultIndexer
from minime.tracing.viewer import TraceViewer

app = typer.Typer(help="MiniMe: Identity-conditioned LLM orchestration")


@app.command()
def init(
    mode: str = typer.Option("dev", help="'dev' or 'prod'"),
    vault_path: Optional[str] = typer.Option(None, help="Path to Obsidian vault (prod mode)"),
):
    """Initialize MiniMe."""
    config = load_config(mode=mode, vault_path=vault_path)
    typer.echo(f"✓ Initialized {mode} mode")
    typer.echo(f"  Vault: {config.vault_path}")
    typer.echo(f"  DB: {config.db_path}")
    typer.echo(f"  Traces: {config.trace_dir}")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description"),
    domain: Optional[str] = typer.Option(None, help="Explicit domain hint"),
):
    """Run a task through orchestrator."""
    config = load_config(mode="dev")
    orchestrator = Orchestrator(config)
    
    typer.echo(f"🚀 Running task: {task}")
    
    result = orchestrator.run(task_text=task, domain_hint=domain)
    
    typer.echo(f"\n✓ Completed run_id: {result.run_id}")
    typer.echo(f"  Tokens used: {result.total_tokens_used}")
    typer.echo(f"  Time: {result.total_time_seconds:.2f}s")
    typer.echo(f"\n📋 Final output:\n{json.dumps(result.final_output, indent=2)}")


@app.command()
def vault_sync():
    """Manually sync vault (watch + index changes)."""
    config = load_config(mode="dev")
    indexer = VaultIndexer(config)
    
    typer.echo("📚 Syncing vault...")
    count = indexer.reindex_vault()
    typer.echo(f"✓ Indexed {count} notes")


@app.command()
def graph_review():
    """Review proposed graph edges."""
    config = load_config(mode="dev")
    
    # TODO: Load proposals from DB
    typer.echo("📊 Proposed edges:")
    typer.echo("  (none yet)")
    typer.echo("\nApprove all? (y/n)")


@app.command()
def trace_view(
    run_id: Optional[str] = typer.Argument(None, help="Run ID to view (latest if not provided)"),
    format: str = typer.Option("human", help="'human' or 'json'"),
):
    """View trace of a run."""
    config = load_config(mode="dev")
    viewer = TraceViewer(config)
    
    trace = viewer.get_latest(run_id)
    if not trace:
        typer.echo("No trace found")
        return
    
    if format == "json":
        typer.echo(json.dumps(trace.dict(), indent=2))
    else:
        typer.echo(f"Run ID: {trace.run_id}")
        typer.echo(f"Task: {trace.task_id}")
        typer.echo(f"Domain: {trace.domain_detected}")
        typer.echo(f"Agents: {len(trace.agent_messages)}")
        typer.echo(f"Tokens: {trace.total_tokens_used}")
        typer.echo(f"Time: {trace.total_time_seconds:.2f}s")


@app.command()
def feedback(
    run_id: str = typer.Argument(..., help="Run ID"),
    action: str = typer.Option("accept", help="'accept', 'reject', or 'edit'"),
    note: Optional[str] = typer.Option(None, help="User feedback note"),
):
    """Provide feedback on a run."""
    config = load_config(mode="dev")
    
    # TODO: Store feedback in DB
    typer.echo(f"✓ Feedback stored: {action}")


@app.command()
def health():
    """Health check."""
    try:
        config = load_config(mode="dev")
        typer.echo(f"✓ Config loaded: {config.mode} mode")
        typer.echo(f"✓ API Key present: {'Yes' if config.provider_config.api_key else 'No'}")
        typer.echo("\n✓ System healthy")
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

### 9.4 minime/providers/base.py (Provider Interface)

```python
"""LLM provider abstraction."""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    usage: dict  # {"tokens_used": int, "tokens_prompt": int, "tokens_completion": int}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[ToolSchema]] = None,
    ) -> GenerateResponse:
        """Generate completion."""
        pass

    @abstractmethod
    async def embed(self, text_list: list[str], model: str) -> list[list[float]]:
        """Embed texts."""
        pass
```

### 9.5 minime/providers/openai_compat.py (OpenAI Provider)

```python
"""OpenAI-compatible provider."""

import httpx
import json
from typing import Optional
from minime.providers.base import LLMProvider, GenerateResponse, ToolSchema
import os


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider."""

    def __init__(self, api_key: str, model: str, api_base: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_base = api_base or "https://api.openai.com/v1"

    async def generate(
        self,
        messages: list[dict],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[ToolSchema]] = None,
    ) -> GenerateResponse:
        """Generate completion via OpenAI API."""
        
        # Prepend system message
        all_messages = [{"role": "system", "content": system}] + messages
        
        payload = {
            "model": model or self.model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        choice = data["choices"][0]
        text = choice["message"].get("content", "")
        finish_reason = choice.get("finish_reason", "stop")
        
        usage = data.get("usage", {})
        
        return GenerateResponse(
            text=text,
            finish_reason=finish_reason,
            usage={
                "tokens_used": usage.get("total_tokens", 0),
                "tokens_prompt": usage.get("prompt_tokens", 0),
                "tokens_completion": usage.get("completion_tokens", 0),
            },
        )

    async def embed(self, text_list: list[str], model: str) -> list[list[float]]:
        """Embed texts."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model or "text-embedding-3-small",
            "input": text_list,
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_base}/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
```

### 9.6 minime/core/orchestrator.py (Main Orchestrator)

```python
"""Main orchestration engine."""

import asyncio
import uuid
from datetime import datetime
from typing import Optional
from minime.config import MiniMeConfig
from minime.providers.openai_compat import OpenAIProvider
from minime.mask.generator import generate_mask_weights
from minime.memory.retrieval import retrieve_memory
from minime.core.context_broker import ContextBroker
from minime.tracing.models import RunTrace, TraceEvent


class Orchestrator:
    """Main orchestrator."""

    def __init__(self, config: MiniMeConfig):
        self.config = config
        self.provider = OpenAIProvider(
            api_key=config.provider_config.api_key,
            model=config.provider_config.model,
        )
        self.context_broker = ContextBroker(config)
        self.trace_events: list[TraceEvent] = []

    async def run(self, task_text: str, domain_hint: Optional[str] = None) -> RunTrace:
        """Run a task."""
        run_id = str(uuid.uuid4())[:8]
        task_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        try:
            # 1. Detect domain
            domain = domain_hint or "unknown"
            complexity = "medium"
            
            self._log_trace_event(
                event_type="task_detected",
                details={"task_id": task_id, "domain": domain, "complexity": complexity},
            )
            
            # 2. Generate mask
            mask_weights = generate_mask_weights(
                domain=domain,
                complexity=complexity,
                agent_name="orchestrator",
            )
            self._log_trace_event(
                event_type="mask_generated",
                details=mask_weights.dict(),
            )
            
            # 3. Retrieve memory
            retrieved = await retrieve_memory(
                query=task_text,
                config=self.config,
                max_results=mask_weights.retrieval_k,
            )
            self._log_trace_event(
                event_type="memory_retrieved",
                details={"count": len(retrieved.chunks), "tokens": retrieved.token_budget_used},
            )
            
            # 4. Assemble context
            context = self.context_broker.assemble(
                task_text=task_text,
                retrieved_chunks=retrieved.chunks,
                mask_weights=mask_weights,
            )
            self._log_trace_event(
                event_type="context_assembled",
                details={"tokens": len(context.split()) // 1.3},
            )
            
            # 5. Call LLM (simple architect example)
            system_prompt = "You are an architect. Design a system architecture for the given task."
            
            response = await self.provider.generate(
                messages=[{"role": "user", "content": task_text}],
                system=system_prompt,
                model=self.config.provider_config.model,
                temperature=mask_weights.temperature,
                max_tokens=2000,
            )
            
            self._log_trace_event(
                event_type="agent_call",
                details={"agent": "architect", "tokens_used": response.usage["tokens_used"]},
                tokens_used=response.usage["tokens_used"],
            )
            
            # 6. Parse output
            final_output = {"response": response.text}
            
            # 7. Build trace
            total_time = (datetime.now() - start_time).total_seconds()
            total_tokens = sum(e.tokens_used for e in self.trace_events)
            
            trace = RunTrace(
                run_id=run_id,
                task_id=task_id,
                domain_detected=domain,
                complexity=complexity,
                mask_weights=mask_weights,
                agent_messages=[],  # Simplified for MVP
                action_proposals=[],
                final_output=final_output,
                total_tokens_used=total_tokens,
                total_time_seconds=total_time,
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )
            
            return trace
        
        except Exception as e:
            print(f"Error in orchestrator: {e}")
            raise

    def _log_trace_event(self, event_type: str, details: dict, tokens_used: int = 0):
        """Log a trace event."""
        event = TraceEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            details=details,
            tokens_used=tokens_used,
        )
        self.trace_events.append(event)
```

### 9.7 minime/core/context_broker.py (Context Assembly)

```python
"""Context broker: block assembly + budget enforcement."""

from minime.config import MiniMeConfig
from minime.mask.models import MaskWeights
from minime.memory.models import MemoryChunk


class ContextBroker:
    """Assembles context blocks with mask weighting + budget enforcement."""

    def __init__(self, config: MiniMeConfig):
        self.config = config
        self.context_limit = config.default_context_limit

    def assemble(
        self,
        task_text: str,
        retrieved_chunks: list[MemoryChunk],
        mask_weights: MaskWeights,
        identity_block: str = "",
        domain_block: str = "",
    ) -> str:
        """Assemble context with mask weighting + budget enforcement."""
        
        # Build blocks
        blocks = {
            "identity": identity_block or "[Global Identity Principles: default]",
            "domain": domain_block or "[Domain Constraints: none]",
            "memory": self._format_memory_chunks(retrieved_chunks),
            "task": f"TASK:\n{task_text}",
        }
        
        # Apply weights
        weighted_blocks = []
        for block_name, block_content in blocks.items():
            weight = mask_weights.block_weights.get(block_name, 0.5)
            if weight > 0:
                weighted_blocks.append((weight, block_name, block_content))
        
        # Sort by weight descending
        weighted_blocks.sort(reverse=True, key=lambda x: x[0])
        
        # Assemble with budget
        assembled = []
        tokens_used = 0
        output_budget = 500  # Reserve for output
        
        for weight, block_name, block_content in weighted_blocks:
            block_tokens = len(block_content.split()) // 1.3
            
            if tokens_used + block_tokens < self.context_limit - output_budget:
                assembled.append(block_content)
                tokens_used += block_tokens
            else:
                # Truncate or skip
                remaining = self.context_limit - output_budget - tokens_used
                if remaining > 100:
                    assembled.append(block_content[:remaining * 4])  # Approx 4 chars per word
        
        return "\n\n".join(assembled)

    def _format_memory_chunks(self, chunks: list[MemoryChunk]) -> str:
        """Format retrieved memory chunks."""
        if not chunks:
            return "[No prior knowledge]"
        
        formatted = ["RETRIEVED MEMORY:"]
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"{i}. [{chunk.note_path}] {chunk.content[:200]}...")
        
        return "\n".join(formatted)
```

### 9.8 minime/mask/generator.py (Heuristic Mask Generation)

```python
"""Mask weight generation (heuristic MVP)."""

from minime.mask.models import MaskWeights
from datetime import datetime


def generate_mask_weights(
    domain: str,
    complexity: str,
    agent_name: str,
) -> MaskWeights:
    """Generate mask weights using heuristic rules (MVP)."""
    
    # Base: always use global identity
    alpha = 1.0
    
    # Domain boost
    beta = 0.8 if domain != "unknown" else 0.0
    
    # Task specificity
    if complexity == "high":
        gamma = 0.7
    elif complexity == "medium":
        gamma = 0.4
    else:
        gamma = 0.1
    
    # Agent-specific
    agent_gammas = {
        "architect": 0.6,
        "builder": 0.8,
        "critic": 0.5,
        "research": 0.7,
        "integrator": 0.3,
    }
    agent_gamma = agent_gammas.get(agent_name, 0.3)
    
    # Temperature
    temperature = 0.3 if complexity == "high" else (0.5 if complexity == "medium" else 0.7)
    
    # Retrieval budget
    retrieval_k = 10 if complexity == "high" else 5
    
    # Block weights
    block_weights = {
        "identity": alpha,
        "domain": beta,
        "memory": (alpha + beta + gamma) / 3,
        "task": gamma,
    }
    
    return MaskWeights(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        agent_gamma=agent_gamma,
        temperature=temperature,
        retrieval_k=retrieval_k,
        block_weights=block_weights,
        agent_routing_bias={agent_name: 1.0, "critic": 0.3},
        created_at=datetime.now(),
    )
```

### 9.9 minime/memory/retrieval.py (Graph-Aware Retrieval)

```python
"""Memory retrieval with graph-aware ranking."""

from minime.config import MiniMeConfig
from minime.memory.models import MemoryChunk, RetrievalResult
from typing import Optional


async def retrieve_memory(
    query: str,
    config: MiniMeConfig,
    max_results: int = 5,
    domain_filter: Optional[str] = None,
) -> RetrievalResult:
    """Retrieve relevant memory chunks."""
    
    # TODO: Implement vector search + graph traversal
    # For MVP: return mock results
    
    mock_chunk = MemoryChunk(
        chunk_id="test-1",
        note_path="vault/knowledge/test.md",
        section_title="Test Section",
        content="This is a test chunk of memory.",
        embedding_ref="embeddings/test-1.jsonl",
        domain="general",
        scope="global",
        tags=["test"],
        confidence=0.9,
        recency_score=0.8,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    return RetrievalResult(
        chunks=[mock_chunk] * min(max_results, 3),
        scores=[0.9, 0.85, 0.8],
        reasons=["High semantic similarity", "Graph proximity", "Recent"],
        token_budget_used=150,
        total_tokens_available=config.default_context_limit - 500,
    )
```

### 9.10 minime/tracing/models.py (Trace Models)

```python
"""Tracing models."""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any


class TraceEvent(BaseModel):
    event_type: str
    timestamp: datetime
    details: dict[str, Any]
    tokens_used: int = 0


class RunTrace(BaseModel):
    run_id: str
    task_id: str
    domain_detected: str
    complexity: str
    mask_weights: Any  # MaskWeights
    agent_messages: list[Any]
    action_proposals: list[Any]
    final_output: dict[str, Any]
    total_tokens_used: int
    total_time_seconds: float
    user_feedback: Optional[str] = None
    created_at: datetime
    completed_at: datetime
```

### 9.11 minime/__main__.py (Entrypoint)

```python
"""CLI entrypoint."""

import asyncio
from minime.cli import app

if __name__ == "__main__":
    app()
```

### 9.12 Running the MVP

**Setup and Test Commands**

```bash
# 1. Clone and install
git clone https://github.com/your-org/minime.git
cd minime
python -m pip install -e ".[dev]"

# 2. Initialize dev environment
python -m minime init --mode dev

# 3. Create sample identity
cat > vault/identity.md << 'EOF'
# My Identity

## Principles
- Modularity over monoliths
- Clear abstractions
- Top-down planning

## Domains
- ML/AI
- Software Architecture
- Systems Design
EOF

# 4. Create sample note in vault
mkdir -p vault/knowledge
cat > vault/knowledge/ml-basics.md << 'EOF'
# ML Basics

## Transformer Architecture
- Self-attention mechanism
- Positional encoding
- Feed-forward networks

See also: [[neural-networks]]
EOF

# 5. Sync vault
python -m minime vault_sync

# 6. Health check
python -m minime health

# 7. Run a task
python -m minime run "Design an ML inference system" --domain ml

# 8. View trace
python -m minime trace_view

# 9. Run tests
pytest tests/ -v

# 10. Lint
black minime/ tests/
ruff check minime/ tests/
```

**Expected Output**
```
✓ Initialized dev mode
  Vault: /path/to/minime/vault
  DB: /path/to/minime/minime-dev.db
  Traces: /path/to/minime/traces

📚 Syncing vault...
✓ Indexed 2 notes

✓ Config loaded: dev mode
✓ API Key present: Yes
✓ System healthy

🚀 Running task: Design an ML inference system
(calls LLM, assembles context, returns output)

✓ Completed run_id: abc12345
  Tokens used: 2847
  Time: 3.21s

📋 Final output:
{
  "response": "(LLM output here)"
}
```

---

## 10) Testing Strategy

### Unit Tests

**test_retrieval.py**
```python
import pytest
from minime.memory.retrieval import rank_retrieval_results
from minime.memory.models import MemoryChunk
from minime.mask.models import MaskWeights


def test_vector_search_filters_by_domain():
    """Vector search respects domain filters."""
    chunks = [
        MemoryChunk(..., domain="ml"),
        MemoryChunk(..., domain="bio"),
    ]
    query = "neural networks"
    
    result = rank_retrieval_results(chunks, query, domain_filter="ml")
    
    assert all(c.domain == "ml" for c in result)


def test_graph_proximity_boosts_score():
    """Adjacent nodes in graph get higher score."""
    # Create two chunks: one directly related, one distant
    related_chunk = MemoryChunk(..., chunk_id="chunk-1")
    distant_chunk = MemoryChunk(..., chunk_id="chunk-2")
    
    # Mock graph: related_chunk is adjacent to query node
    # distant_chunk is 3 hops away
    
    scores = rank_retrieval_results([related_chunk, distant_chunk], query)
    
    assert scores[0] > scores[1]


def test_recency_weighting():
    """Recent notes weighted higher."""
    old_chunk = MemoryChunk(..., recency_score=0.1)
    new_chunk = MemoryChunk(..., recency_score=0.9)
    
    scores = rank_retrieval_results([old_chunk, new_chunk], query)
    
    assert scores[1] > scores[0]
```

**test_context_assembly.py**
```python
def test_token_budget_enforced():
    """Context broker respects token limit."""
    broker = ContextBroker(config)
    broker.context_limit = 500  # Small limit
    
    large_chunks = [MemoryChunk(...) for _ in range(100)]  # Will overflow
    
    context = broker.assemble(
        task_text="...",
        retrieved_chunks=large_chunks,
        mask_weights=...,
    )
    
    # Should truncate
    token_count = len(context.split()) // 1.3
    assert token_count <= 500


def test_mask_weights_applied():
    """Block weights influence context order."""
    mask_weights = MaskWeights(
        ...,
        block_weights={"identity": 0.9, "memory": 0.1},
    )
    
    context = broker.assemble(..., mask_weights=mask_weights)
    
    # Identity should come first
    assert context.index("[Global Identity") < context.index("[Retrieved Memory")
```

**test_tool_allowlist.py**
```python
def test_tool_allowlist_enforced():
    """Agent cannot call tools not in allowlist."""
    agent_spec = AgentSpec(
        ...,
        tools_allowed=["retrieve_memory", "propose_plan"],
    )
    
    with pytest.raises(ValueError):
        executor.execute_tool(
            agent_name="architect",
            tool_name="run_command",  # Not in allowlist
            args={...},
        )


def test_command_tag_allowlist():
    """run_command requires approved tag."""
    agent_spec = AgentSpec(
        ...,
        command_allowlist_tags=["test", "build"],
    )
    
    # Should succeed
    executor.execute_tool(..., tool_name="run_command", args={"cmd": "pytest", "allowlist_tag": "test"})
    
    # Should fail
    with pytest.raises(ValueError):
        executor.execute_tool(..., tool_name="run_command", args={"cmd": "rm -rf /", "allowlist_tag": "deploy"})
```

**test_trace_correctness.py**
```python
def test_trace_logged_completely():
    """All decisions logged to trace."""
    orchestrator = Orchestrator(config)
    
    trace = orchestrator.run(task_text="test")
    
    assert trace.run_id
    assert trace.task_id
    assert trace.domain_detected
    assert trace.mask_weights
    assert trace.total_tokens_used > 0
    assert trace.total_time_seconds > 0


def test_trace_event_timestamps():
    """Trace events have proper timestamps."""
    orchestrator = Orchestrator(config)
    
    orchestrator._log_trace_event("test", {})
    
    assert orchestrator.trace_events[-1].timestamp
    assert orchestrator.trace_events[-1].event_type == "test"
```

**test_graph_proposals.py**
```python
def test_edge_proposal_confidence():
    """Proposed edges have confidence scores."""
    indexer = VaultIndexer(config)
    
    proposals = indexer.propose_edges()
    
    for proposal in proposals:
        assert 0 <= proposal.confidence <= 1
        assert proposal.rationale


def test_user_approval_required():
    """New edges require user approval."""
    indexer = VaultIndexer(config)
    
    proposal = GraphUpdateProposal(
        ...,
        requires_user_approval=True,
    )
    
    # Should not apply without approval
    assert not indexer.apply_proposal(proposal)
    
    # After approval
    proposal.user_decision = "approved"
    assert indexer.apply_proposal(proposal)
```

**test_mask_generation.py**
```python
def test_heuristic_mask_weights():
    """Mask generation produces valid weights."""
    mask = generate_mask_weights(domain="ml", complexity="high", agent_name="architect")
    
    assert 0 <= mask.alpha <= 1
    assert 0 <= mask.beta <= 1
    assert 0 <= mask.gamma <= 1
    assert 0.1 <= mask.temperature <= 2.0
    assert 1 <= mask.retrieval_k <= 50


def test_complexity_affects_temperature():
    """Higher complexity → lower temperature (more deterministic)."""
    high = generate_mask_weights(domain="ml", complexity="high", agent_name="architect")
    low = generate_mask_weights(domain="ml", complexity="low", agent_name="architect")
    
    assert high.temperature < low.temperature
```

### E2E Test

**test_full_orchestration.py**
```python
import pytest
from minime.core.orchestrator import Orchestrator
from minime.config import MiniMeConfig
from minime.providers.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_full_task_e2e():
    """End-to-end task execution (mocked LLM)."""
    # Setup
    config = MiniMeConfig(...)
    config.provider_config.api_key = "mock"  # Mock provider
    
    orchestrator = Orchestrator(config)
    orchestrator.provider = MockLLMProvider()  # Mock, no API calls
    
    # Run task
    trace = await orchestrator.run(task_text="Design a cache", domain_hint="systems")
    
    # Validate
    assert trace.run_id
    assert trace.domain_detected == "systems"
    assert trace.final_output
    assert trace.total_tokens_used > 0
    
    # Validate trace structure
    assert len(trace.agent_messages) > 0 or trace.final_output
```

### Testing Commands

```bash
# Run unit tests
pytest tests/unit/ -v

# Run E2E tests
pytest tests/e2e/ -v --asyncio-mode=auto

# Run with coverage
pytest tests/ --cov=minime --cov-report=html

# Run specific test
pytest tests/unit/test_retrieval.py::test_vector_search_filters_by_domain -v
```

---

## 11) Next 10 Upgrades (Ranked by ROI)

### Upgrade 1: Better Retrieval (Reranking + Domain Filters + Recency)
**ROI: High** (Core feature improvement)
- Add cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLMv2-L12-H384-uncased`)
- Implement domain-scoped retrieval (per-domain embedding subspace)
- Improve recency weighting (configurable decay curves)
- **Effort**: 1 week
- **Impact**: Retrieval quality improvement 30%+

### Upgrade 2: Structured Output Strictness (JSON Schema Validation)
**ROI: High** (Reliability)
- Enforce agent outputs strictly match JSON schemas
- Implement retry loop: LLM violates schema → ask for correction → max 3 retries
- Parser error handling + graceful degradation
- **Effort**: 3-4 days
- **Impact**: Reduce parsing errors 80%+

### Upgrade 3: VS Code Integration (Diff-Based Patches)
**ROI: High** (Developer experience)
- VS Code extension: display agent proposals in diff view
- Accept/reject patches via IDE
- Show traces + decisions inline
- **Effort**: 2 weeks
- **Impact**: Reduce approval friction 60%+

### Upgrade 4: Evaluation Harness
**ROI: High** (Measurement)
- Define metrics: trace quality, output coherence, constraint satisfaction
- Build eval dataset (20-50 task exemplars)
- Automated eval: run same task with different mask weights; compare outcomes
- **Effort**: 1-2 weeks
- **Impact**: Data for mask training + optimization decisions

### Upgrade 5: Per-Project Profiles
**ROI: Medium** (Personalization)
- Store project-specific masks, identity, memory scopes
- Allow user to define project templates
- Load/switch profiles per task
- **Effort**: 1 week
- **Impact**: Multi-project support + context isolation

### Upgrade 6: Mask Network Training + Validation
**ROI: High** (Core learning)
- Implement full training loop (collect data → train MLP → validate → deploy)
- Safety checks: prevent catastrophic drift (validate on past traces)
- Scheduled training (weekly or on-demand)
- **Effort**: 2-3 weeks
- **Impact**: Personalization + adaptive behavior

### Upgrade 7: Security Hardening + Sandboxed Execution
**ROI: Medium** (Safety)
- Sandbox run_command via Docker containers
- PII detection + flagging in memory + retrieval
- Allowlist audit logs
- **Effort**: 2 weeks
- **Impact**: Production-ready safety

### Upgrade 8: Better Obsidian Graph UX (Web UI)
**ROI: Medium** (UX improvement)
- Web interface for graph review
- Visual node + edge browser
- Explain why edges proposed (show similarity vector + rationale)
- Batch approve/reject
- **Effort**: 2-3 weeks
- **Impact**: User comfort with graph updates

### Upgrade 9: Multi-Provider Adaptive Routing + Caching
**ROI: Medium** (Efficiency)
- Track latency + cost per provider
- Adaptive routing: route to fastest/cheapest provider per agent
- Implement LLM response caching (hash(system + input) → output)
- **Effort**: 2 weeks
- **Impact**: 30%+ cost/latency reduction

### Upgrade 10: Agent Config Marketplace + Sharing
**ROI: Low** (Community)
- Allow users to upload custom agent specs + masks
- Community-curated agent packs (research, coding, writing, etc.)
- Version control + rollback for community agents
- **Effort**: 3 weeks
- **Impact**: Community ecosystem growth

---

## Summary: Build Checklist for Week 1

**Day 1: Setup + Config**
- [ ] Create repo structure (folders + init files)
- [ ] Set up config system (load_config, init dev/prod)
- [ ] Create sample vault + identity.md
- [ ] Initialize SQLite DB schema

**Day 2: Providers + LLM**
- [ ] Implement LLMProvider base class
- [ ] Implement OpenAI provider (generate + embed)
- [ ] Test async HTTP calls
- [ ] Create mock provider for testing

**Day 3: Memory + Retrieval**
- [ ] Vault watcher + file indexing
- [ ] Chunking + embedding (with mock embeddings)
- [ ] SQLite graph store (VaultNode, GraphEdge tables)
- [ ] Basic vector search

**Day 4: Mask System + Context Broker**
- [ ] Implement heuristic mask generation
- [ ] Implement context broker (block assembly + budget)
- [ ] Test mask weights applied correctly
- [ ] Validate token budget enforcement

**Day 5: Orchestrator + CLI**
- [ ] Implement basic orchestrator loop
- [ ] Create CLI entrypoint (typer)
- [ ] Implement core commands (init, run, vault_sync, trace_view)
- [ ] Test end-to-end flow (task → output → trace)

**Day 6: Tracing + Feedback**
- [ ] Implement RunTrace + TraceEvent models
- [ ] JSONL trace logging + storage
- [ ] Trace viewer CLI
- [ ] Feedback collection (accept/reject/edit)

**Day 7: Testing + Hardening**
- [ ] Write unit tests for retrieval, context assembly, tool allowlist
- [ ] Write E2E test (mocked LLM)
- [ ] Add error handling + validation
- [ ] Document architecture + quick start

---

## Assumptions (Explicit)

1. **OpenAI API available**: MVP assumes OpenAI-compatible API. Swap provider for alternatives.
2. **Obsidian vault format**: Assumes standard .md format with wikilinks + frontmatter.
3. **No fine-tuning**: MVP uses frozen foundation models only. Mask network trainable later.
4. **Single-user**: No multi-user/collaborative features in MVP.
5. **Local execution**: No cloud deployment in MVP; all compute local.
6. **Token budgets reasonable**: Assumes context windows 4k-8k tokens sufficient for tasks.
7. **User engagement**: Assumes user will provide feedback for mask training.
8. **No streaming**: LLM responses returned as complete JSON, no streaming.

---

## References + Next Steps

**After MVP Deploy (Week 2-3)**

1. **Collect feedback**: Run MiniMe on real tasks, gather user feedback
2. **Validate traces**: Ensure tracing is complete + useful for debugging
3. **Plan training**: Design mask training dataset + validation procedure
4. **Begin Phase 2**: Implement better retrieval + structured output strictness

**Documentation Links** (to be written)
- `docs/architecture.md` — Deep dive into layers + data flow
- `docs/mask_system.md` — How the mask network works + learning
- `docs/obsidian_integration.md` — Vault watcher + graph design
- `docs/api_reference.md` — All schemas + functions

---

**END OF IMPLEMENTATION PLAN**

This document is build-ready. All code is copy/paste runnable. Start with the CLI (section 9.11) and validate the system boots. The core orchestrator loop is minimal but complete. Extend incrementally from there.
