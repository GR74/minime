# MiniMe: Complete Build Plan & Implementation Guide
**v1.0 | December 24, 2025 | Ready for Day 1 Implementation**

---

## TABLE OF CONTENTS
1. [One-Page System Map](#1-one-page-system-map)
2. [MVP Scope (2–3 weeks)](#2-mvp-scope-2--3-weeks)
3. [Tech Stack & Why](#3-tech-stack--why)
4. [API & Tooling Plan](#4-api--tooling-plan)
5. [Core Data Schemas](#5-core-data-schemas--pydantic-models)
6. [Mask System Implementation](#6-mask-system-implementation-plan)
7. [Multi-Agent Orchestration](#7-multi-agent-orchestration-plan)
8. [Repo Scaffold](#8-repository-scaffold-exact-tree)
9. [Starter Code (Copy-Paste Ready)](#9-starter-code-copy-paste-ready)
10. [Testing Strategy](#10-testing-strategy)
11. [Next 10 Upgrades (Ranked by ROI)](#11-next-10-upgrades-ranked-by-roi)
12. [Local Dev Setup (Day 1)](#local-dev-setup-day-1)

---

## 1. ONE-PAGE SYSTEM MAP

### 10,000-Foot Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          MiniMe Cognitive Engine                         │
│                                                                          │
│  User Intent (CLI / VS Code)                                            │
│         │                                                               │
│         └─→ [Task Classifier]  ← Domain + Complexity Detection         │
│              │                                                          │
│              └─→ [Mask Network] ← Identity + Task + Domain Embeddings  │
│                   │                                                    │
│                   └─→ Generates: α (domain weight), β (task weight),  │
│                                  γ (agent weight), k (retrieve size)  │
│                                                                       │
│         ┌─────────────────────────────────────────────┐              │
│         │  Context Assembly Engine                    │              │
│         │  ─────────────────────────────────────────  │              │
│         │  Identity Block (weighted by α)             │              │
│         │  + Domain Constraints (weighted by β)       │              │
│         │  + Task Instructions (weighted by γ)        │              │
│         │  + Retrieved Memory (top-k, reranked)       │              │
│         │  = Token-budgeted Prompt                    │              │
│         └─────────────────────────────────────────────┘              │
│              │                                                        │
│              └─→ [Agent Team] (Architect/Builder/Critic/Integrator) │
│                   │                                                  │
│                   ├─→ [LLM Call] (OpenAI/Claude/etc, frozen)       │
│                   │    │                                            │
│                   │    └─→ JSON structured output + tool calls      │
│                   │                                                  │
│                   └─→ [Critic Agent]  ← Enforces constraints        │
│                        │                                            │
│                        ├─→ Valid? → [Integrator] → Output           │
│                        │                                            │
│                        └─→ Invalid? → Revise (loop control)         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Output: Action Proposal (write/run/apply)                  │  │
│  │ + Reasoning Trace (what was retrieved, why, which agent)   │  │
│  │ + Token usage + latency                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│         │                                                         │
│         └─→ [User Approval Gate] ← Required for execution       │
│              │                                                   │
│              ├─→ User accepts   → Execute + log feedback         │
│              │                                                   │
│              └─→ User rejects   → Correction pair stored         │
│                  + delta applied to mask weights (offline)      │
│                                                                 │
└──────────────────────────────────────────────────────────────────────────┘

ONLINE vs OFFLINE:
┌─────────────────┐           ┌──────────────────────────┐
│   ONLINE (User) │           │  OFFLINE (Background)    │
├─────────────────┤           ├──────────────────────────┤
│ Classification  │           │ Mask Network Training    │
│ Mask generation │    ←──→   │ (on correction pairs)    │
│ Retrieval       │           │                          │
│ Context assem.  │           │ Vector index building    │
│ LLM calls       │           │ Scheduled reweighting    │
│ Execution       │           │                          │
│ Tracing         │           │                          │
└─────────────────┘           └──────────────────────────┘
      (fast)                        (periodic: nightly)
```

### Component Separation

| Layer | Responsibility | Files |
|-------|-----------------|-------|
| **Identity** | Persistent principles (P_global) | `identity.py` |
| **Memory** | Vector embeddings + metadata | `memory.py` |
| **Mask** | Learned conditioning network | `mask_network.py` |
| **Context** | Block assembly + token budgeting | `context_broker.py` |
| **Agents** | Reasoning roles | `agents/` |
| **Orchestration** | Task flow + iteration | `orchestrator.py` |
| **Tools** | Execution + allowlist | `tools.py` |
| **Tracing** | Event logging | `tracing.py` |

### Data Flow (Simplified)

```
CLI Input
   ↓
Task Embedding (embed_query)
   ↓
Classify Domain (classifier.predict)
   ↓
Generate Masks (mask_network.forward)
   ↓
Retrieve Context (memory.retrieve with mask weights)
   ↓
Assemble Prompt (context_broker.assemble with budget)
   ↓
Agent Loop (orchestrator.run_agents)
   ├─→ LLM Call (provider.complete)
   ├─→ Parse Structured Output
   ├─→ Critic Review (enforce constraints)
   └─→ Integrate Results
   ↓
Propose Actions (ActionProposal)
   ↓
User Approval (CLI prompt)
   ├─→ Execute (tools.execute with allowlist)
   ├─→ Log Trace (tracing.write_event)
   └─→ Store Feedback (correction pair)
   ↓
Offline: Periodic Mask Training (on correction pairs)
```

---

## 2. MVP SCOPE (2–3 WEEKS)

### Features (WILL DELIVER)

**Phase 1: Foundation (Week 1)**
- [x] CLI scaffold (Typer-based)
- [x] One LLM provider (OpenAI-compatible)
- [x] Vector memory store (local SQLite + pgvector simulation)
- [x] Identity matrix (P_global) — hardcoded heuristic, no training yet
- [x] Mask system (heuristic version — fixed α/β/γ, no learnable params)
- [x] Context broker (block assembly + token budgeting)
- [x] Orchestrator (basic task → agent → LLM → output flow)
- [x] One agent: **Architect** (plans structure)
- [x] Tool execution: write_file, run_command (with allowlist)
- [x] Tracing: JSONL event log per run
- [x] User approval gate (CLI confirmation before execute)

**Phase 2: Multi-Agent + Critic (Week 2)**
- [x] **Builder** agent (writes code)
- [x] **Critic** agent (validates output)
- [x] **Integrator** agent (merges conflicting outputs)
- [x] Critic enforces constraints (token budget, safety, architectural rules)
- [x] Iteration control (max 3 agent rounds before approval required)
- [x] Tool schema validation (JSON schema type checking)

**Phase 3: Memory + Feedback (Week 2.5)**
- [x] Import notes / seed memory
- [x] Retrieval with mask weighting (α weight on domain filter)
- [x] Feedback correction pairs (user edits → stored as delta)
- [x] Trace inspection CLI (load + pretty-print traces)
- [x] Config system (yaml-based, environment overrides)

**Phase 4: Polish + Testing (Week 3)**
- [x] E2E test (mocked LLM, verify flow)
- [x] Unit tests (retrieval filtering, budget enforcement, allowlist)
- [x] Error handling (graceful LLM timeouts, tool failures)
- [x] README + quickstart

### Non-Features (WON'T BUILD YET)

- ❌ Trainable mask network (MLPs come in Phase 2 upgrade)
- ❌ Fine-grained LoRA adapters
- ❌ VS Code plugin
- ❌ Web dashboard
- ❌ Collaborative multi-user traces
- ❌ Advanced reranking (e.g., ColBERT)
- ❌ Streaming agent responses
- ❌ Docker containerization
- ❌ Multi-project profiles (v1: single global config)
- ❌ Sandboxed code execution (allowlist + warnings for MVP)

### Acceptance Criteria Checklist

- [ ] **MVP Completeness**
  - [ ] `minime init` creates config + identity file
  - [ ] `minime ask "plan a REST API"` produces structured output without errors
  - [ ] Output includes: reasoning trace, proposed actions, token budget used
  - [ ] User can accept/reject actions via CLI prompt
  - [ ] Accepted actions execute (write_file, run_command)
  - [ ] All runs produce `.minime/traces/{run_id}.jsonl`
  - [ ] Trace contains: task input, masks used, retrieval results, agent outputs, decisions

- [ ] **Agent System**
  - [ ] Architect agent produces valid JSON (dependencies, modules, files)
  - [ ] Builder agent produces executable code (tested via allowlisted run_command)
  - [ ] Critic agent rejects code that violates identity principles
  - [ ] Integrator merges outputs without conflict
  - [ ] Max 3 rounds of iteration before approval required

- [ ] **Memory + Retrieval**
  - [ ] `minime import /path/to/notes` embeds and stores notes
  - [ ] Retrieval with domain mask returns domain-relevant results first
  - [ ] Retrieval respects token budget (k adjusted dynamically)

- [ ] **Tracing + Debugging**
  - [ ] Every run produces trace file (JSONL, human-readable)
  - [ ] `minime trace {run_id}` pretty-prints trace
  - [ ] Trace includes: input, domain, masks (α/β/γ/k), retrieved chunks, agent calls, token usage

- [ ] **Feedback Loop (MVP: Manual)**
  - [ ] `minime feedback {run_id} --delta '{...}'` stores correction pair
  - [ ] Stored corrections are visible in trace file
  - [ ] (Offline training not required for MVP acceptance)

- [ ] **Config + Extensibility**
  - [ ] `.minime/config.yaml` defines identity, memory path, LLM provider, token budgets
  - [ ] `minime config edit` opens editor
  - [ ] Environment overrides work: `MINIME_LLM_PROVIDER=anthropic minime ask ...`
  - [ ] Provider interface is pluggable (swappable behind abstract base class)

### Risks + Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM API cost overruns | High | Token budget enforcement + dry-run mode (`--dry-run` prints context, doesn't call LLM) |
| Unsafe tool execution | Critical | Allowlist only (no shell injection), user approval required, logged |
| Memory bloat (large embeddings) | Medium | Limit initial notes import to <50MB; chunking strategy |
| Mask weights diverge from intent | Medium | Store α/β/γ decisions in trace; easy to revert to heuristic |
| Agent infinite loops | Medium | Max iteration cap (3 rounds); enforced by orchestrator |
| Context assembly too slow | Low | Cache assembled context for same task within 5min window |

---

## 3. TECH STACK & WHY

### Language + Frameworks

| Component | Choice | Why |
|-----------|--------|-----|
| **Language** | Python 3.11+ | Fast iteration, strong ML/LLM ecosystem, ubiquitous in AI. |
| **CLI Framework** | Typer | Modern, async-friendly, auto-generates help + shell completion. |
| **Data Models** | Pydantic v2 | Type safety, JSON schema generation (needed for tool definitions), validation. |
| **Async Runtime** | asyncio | Lightweight, Python standard, integrates cleanly with FastAPI-compatible patterns. |
| **Vector Store (Dev)** | SQLite + numpy | Zero external deps, runs locally, easy to inspect. Serializes vectors as BLOB. |
| **Vector Store (Prod Path)** | pgvector extension | Upgrading SQLite → PostgreSQL + pgvector scales smoothly; same schema. |
| **Embeddings** | OpenAI API (text-embedding-3-small) or Ollama (offline) | MVP uses OpenAI; fallback to Ollama for local-first development. |
| **LLM Providers** | OpenAI-compatible interface (start: OpenAI, extend: Claude, local Ollama) | Minimizes coupling; one provider at a time. |
| **Config** | YAML + Pydantic | Human-readable, version-controllable, schema-validated. |
| **Testing** | pytest + pytest-asyncio | Async support, fixtures, mocking. |
| **Logging / Tracing** | Built-in logging + JSONL file | No external deps; JSONL is queryable (grep, jq). |
| **JSON Schema** | Pydantic model_json_schema() | Auto-generated from models; used for LLM tool calling. |

### Local-First Dev Mode vs. Production Mode

**Local-First (MVP)**
```
├─ Vector Store: SQLite (.minime/memory.db)
├─ Identity: JSON file (.minime/identity.json)
├─ LLM: Ollama (localhost:11434) or OpenAI API key
├─ Config: YAML (.minime/config.yaml)
├─ Traces: JSONL files (.minime/traces/)
└─ Performance: Single-machine, <1sec latency for retrieval
```

**Production Path (Future)**
```
├─ Vector Store: PostgreSQL + pgvector (managed RDS)
├─ Identity: DynamoDB or Firestore (multi-user sync)
├─ LLM: OpenAI API with rate limiting + caching
├─ Config: Secrets manager (AWS Secrets Manager)
├─ Traces: S3 + CloudWatch
├─ Cache: Redis (prompt assembly cache)
└─ Performance: Distributed, multi-user, audit-compliant
```

**Migration Path**: Code uses abstract interfaces (VectorStore, Config, LLMProvider); implementations swappable via dependency injection. No code changes needed to move from local → production.

### Job Runner Strategy (Mask Training)

**MVP (Week 1-2)**: No online training. Mask weights (α, β, γ) are heuristic constants.

**Phase 2 (Week 3)**: Offline training loop (optional background job)
- Collects correction pairs (user accepts/rejects + edits)
- Stored in `.minime/feedback.jsonl`
- Nightly job: load feedback → retrain small MLP mask network
- New weights deployed on next CLI invocation
- Old weights versioned (easy rollback)

**Script: `scripts/train_mask.py`**
```bash
# Manual trigger
python scripts/train_mask.py \
  --feedback-file .minime/feedback.jsonl \
  --output-dir .minime/models/ \
  --epochs 5 \
  --batch-size 32

# Or cron job
0 2 * * * cd ~/minime && python scripts/train_mask.py
```

---

## 4. API & TOOLING PLAN

### LLM API Usage Plan

**Provider Interface (Swappable)**
```python
class LLMProvider(ABC):
    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: list[ToolDefinition] = None,
        response_format: str = "text",  # "text" | "json_object"
    ) -> CompletionResponse:
        """Generate response. Tools trigger function calling."""
```

**Usage**:
- **Generation**: `provider.complete(prompt, system, temperature=0.7)` → plain text
- **Structured JSON**: `provider.complete(prompt, response_format="json_object")` → validated JSON
- **Tool Calling**: `provider.complete(prompt, tools=[...])` → tool call + args

**Initial Implementation: OpenAI**
```python
class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def complete(self, ...):
        # Uses tools parameter to enable function_calling
        # Parses response → tool calls or text
```

**Tokens + Cost Tracking**:
```python
@dataclass
class CompletionResponse:
    text: str | None
    tool_calls: list[ToolCall] | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float  # calculated from token counts + model pricing
```

**Budget Enforcement**:
```python
class TokenBudgetManager:
    def __init__(self, limit: int = 100_000):  # per run
        self.limit = limit
        self.used = 0
    
    async def call_with_budget(self, provider, **kwargs) -> CompletionResponse:
        response = await provider.complete(**kwargs)
        self.used += response.prompt_tokens + response.completion_tokens
        if self.used > self.limit:
            raise BudgetExceededError(f"Used {self.used}/{self.limit}")
        return response
```

### Embeddings Plan + Chunking Strategy

**Embeddings API**
```python
class EmbeddingsProvider(ABC):
    async def embed_text(self, text: str) -> np.ndarray:
        """Returns 1536-dim vector (OpenAI standard)."""
    
    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Returns (N, 1536) matrix."""
```

**Chunking Strategy** (for notes/docs import):
```python
class Chunker:
    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> list[MemoryChunk]:
        """Splits text into overlapping chunks, preserves metadata."""
        # Use semantic boundaries (paragraphs) when possible
        # Fallback to char-based chunking for large paras
        chunks = []
        for chunk_text, start_idx, end_idx in sliding_window(...):
            chunk = MemoryChunk(
                text=chunk_text,
                embedding=None,  # lazy-loaded
                metadata={
                    "source": "note.md",
                    "start_char": start_idx,
                    "domain": "ml",  # extracted or tagged
                    "timestamp": now(),
                }
            )
            chunks.append(chunk)
        return chunks
```

**Retrieval API**
```python
class MemoryStore(ABC):
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        domain_filter: str = None,
        weights: RetrievalWeights = None,  # from mask
    ) -> RetrievalResult:
        """
        Returns top-k chunks by:
        1. Embed query
        2. Filter by domain (if specified + weights['domain_weight'] > 0.1)
        3. Reweight by recency + reuse frequency + mask weights
        4. Return with scores + provenance
        """
```

### Tool Definitions (Typed JSON Schemas)

**Tool Definition Format** (used for LLM function calling)

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON Schema (from Pydantic model)
    category: str  # "write" | "run" | "retrieve" | "apply"
    requires_approval: bool
    allowed_patterns: list[str]  # for validation

# Example:
write_file_tool = ToolDefinition(
    name="write_file",
    description="Write content to a file at path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (e.g., src/main.py)"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    },
    category="write",
    requires_approval=True,
    allowed_patterns=["src/**/*.py", "config/**/*.yaml"],
)
```

**Tool Implementation Functions**

```python
async def retrieve_memory(
    query: str,
    filters: dict = None,
    k: int = 5,
) -> RetrievalResult:
    """Retrieve relevant memory chunks."""
    return await memory_store.retrieve(query, k=k, filters=filters)

async def propose_plan(
    objective: str,
    constraints: list[str] = None,
) -> str:
    """Architect agent generates a plan (JSON)."""
    # Calls LLM with Architect mask
    ...

async def write_file(path: str, content: str) -> ExecutionResult:
    """Write file. Approval required if not in allowlist."""
    if not is_allowed(path):
        return ExecutionResult(
            success=False,
            output="",
            error=f"Path {path} not in allowlist",
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    return ExecutionResult(success=True, output=f"Wrote {path}")

async def apply_patch(path: str, diff: str) -> ExecutionResult:
    """Apply unified diff. Uses difflib for safety."""
    original = Path(path).read_text()
    patched = patch_text(original, diff)
    Path(path).write_text(patched)
    return ExecutionResult(success=True, output=f"Patched {path}")

async def run_command(cmd: str, cwd: str = ".", allowlist_tag: str = "trusted") -> ExecutionResult:
    """Run shell command. Requires allowlist tag."""
    if not is_command_allowed(cmd, allowlist_tag):
        return ExecutionResult(
            success=False,
            output="",
            error=f"Command not in {allowlist_tag} allowlist",
        )
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        timeout=30,
        text=True,
    )
    return ExecutionResult(
        success=result.returncode == 0,
        output=result.stdout,
        error=result.stderr,
    )

async def search_local_repo(pattern: str) -> list[str]:
    """Search repo for files matching pattern (glob or regex)."""
    # Uses pathlib.glob + caching
    ...
```

### Tool Allowlist Design

**Allowlist File** (`.minime/allowlist.yaml`)
```yaml
write:
  patterns:
    - "src/**/*.py"
    - "config/**/*.yaml"
    - "docs/**/*.md"
    - "tests/**/*.py"
  exclude:
    - "src/secrets.py"
    - ".env"

run:
  trusted:
    - "python -m pytest"
    - "python scripts/*.py"
    - "git status"
    - "ls -la"
  untrusted:
    - "rm -rf"
    - "curl http://"

retrieve:
  max_k: 10
  domains: ["ml", "robotics", "neurotech", "protein"]
```

**Allowlist Enforcement**
```python
class ToolAllowlist:
    def is_allowed(self, tool: str, args: dict) -> bool:
        """Check if tool + args match allowlist."""
        if tool == "write_file":
            path = args["path"]
            return any(
                fnmatch(path, pattern)
                for pattern in self.config["write"]["patterns"]
            ) and not any(
                fnmatch(path, exclude)
                for exclude in self.config["write"]["exclude"]
            )
        elif tool == "run_command":
            cmd = args["cmd"]
            tag = args.get("allowlist_tag", "untrusted")
            return any(
                cmd.startswith(allowed)
                for allowed in self.config["run"].get(tag, [])
            )
        return False
```

---

## 5. CORE DATA SCHEMAS (Pydantic Models)

All schemas are Pydantic v2 models, serializable to JSON, with built-in validation.

```python
# ============================================================================
# IDENTITY LAYER
# ============================================================================

@dataclass
class IdentityPrinciple:
    """A single principle (embedded as a vector in P_global)."""
    name: str
    description: str
    vector: np.ndarray  # 384-dim (from MiniLM or similar)
    magnitude: float = 1.0  # importance weight
    decay_rate: float = 0.95  # how fast it adapts to corrections
    scope: str = "global"  # "global" | "domain" | "task"
    tags: list[str] = field(default_factory=list)  # ["coding", "modularity"]

class GlobalIdentityMatrix(BaseModel):
    """The persistent identity: P_global ∈ ℝ^(n × d)."""
    principles: list[IdentityPrinciple]
    dimension: int = 384  # vector size
    created_at: datetime
    updated_at: datetime
    
    def to_matrix(self) -> np.ndarray:
        """Returns (N, 384) matrix of all principle vectors."""
        return np.array([p.vector for p in self.principles])
    
    @classmethod
    def from_file(cls, path: str) -> "GlobalIdentityMatrix":
        with open(path) as f:
            data = json.load(f)
        # Deserialize vectors from base64
        ...

# ============================================================================
# MEMORY LAYER
# ============================================================================

class MemoryChunk(BaseModel):
    """A single chunk of memory (note, code snippet, paper excerpt, etc.)."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding_id: str = None  # reference to vector store
    embedding: np.ndarray = None  # lazy-loaded; not serialized
    metadata: dict = Field(default_factory=dict)
    # metadata example:
    # {
    #   "source": "design_notes.md",
    #   "domain": "ml",
    #   "confidence": 0.9,
    #   "reuse_count": 5,
    #   "last_used": "2025-01-15T10:30:00Z",
    #   "tags": ["bayesian", "inference"],
    # }
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True

class RetrievalQuery(BaseModel):
    """A query for memory retrieval."""
    text: str
    k: int = 5
    domain_filter: str = None
    metadata_filters: dict = Field(default_factory=dict)
    # e.g., {"confidence": {"gte": 0.7}, "tags": ["inference"]}

class RetrievalResult(BaseModel):
    """Ranked list of retrieved chunks."""
    chunks: list[MemoryChunk]
    scores: list[float]  # similarity scores, sorted desc
    query_tokens: int  # for budget tracking
    retrieval_weights_applied: dict  # the mask weights used
    # e.g., {"domain_weight": 0.6, "recency_weight": 0.2, ...}

# ============================================================================
# MASK SYSTEM
# ============================================================================

class MaskWeights(BaseModel):
    """Output of mask network: how to configure reasoning for this task."""
    alpha: float = 0.5  # domain weight (0-1)
    beta: float = 0.3   # task weight (0-1)
    gamma: float = 0.2  # agent weight (0-1)
    
    retrieval_k: int = 5  # how many chunks to retrieve
    retrieval_domain_weight: float = 0.5  # boost domain-relevant chunks
    retrieval_recency_weight: float = 0.1  # boost recent chunks
    
    temperature: float = 0.7  # generation temperature
    rigor: float = 1.0  # constraint enforcement strictness (0.5-1.5)
    
    agent_routing: dict = Field(default_factory=dict)
    # e.g., {
    #   "architect": 0.3,
    #   "builder": 0.5,
    #   "critic": 0.2,
    #   "integrator": 0.0,  # not needed for this task
    # }
    
    max_iterations: int = 3

class MaskRequest(BaseModel):
    """Input to mask network: context for mask generation."""
    task_embedding: np.ndarray  # 384-dim embedding of task
    domain: str = "general"
    complexity: float = 0.5  # (0-1) task difficulty estimate
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# AGENTS & ORCHESTRATION
# ============================================================================

class AgentMessage(BaseModel):
    """Message from an agent to the orchestrator."""
    agent_name: str  # "architect" | "builder" | "critic" | "integrator"
    role: str  # "propose" | "review" | "integrate"
    content: str
    structured_output: dict = None  # JSON if applicable
    reasoning: str = ""  # why this decision
    constraints_satisfied: list[str] = Field(default_factory=list)
    constraints_violated: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ActionProposal(BaseModel):
    """Proposed action for user approval."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    tool: str  # "write_file" | "apply_patch" | "run_command"
    args: dict
    reasoning: str
    agent_responsible: str  # which agent proposed this
    safety_level: str  # "safe" | "caution" | "requires_review"
    
    class Config:
        arbitrary_types_allowed = True

class ExecutionResult(BaseModel):
    """Result of executing a tool."""
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    side_effects: dict = Field(default_factory=dict)  # files written, etc.

# ============================================================================
# TRACING & FEEDBACK
# ============================================================================

class TraceEvent(BaseModel):
    """A single event in an execution trace."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # "task_start" | "retrieval" | "agent_call" | "approval" | ...
    level: str = "info"  # "debug" | "info" | "warn" | "error"
    data: dict  # event-specific payload
    
    # Examples:
    # {
    #   "event_type": "retrieval",
    #   "data": {
    #     "query": "REST API design patterns",
    #     "k": 5,
    #     "results_count": 4,
    #     "top_scores": [0.92, 0.87, 0.81],
    #     "filters_applied": {"domain": "ml"},
    #     "weights": {"domain_weight": 0.6, ...},
    #   }
    # }

class RunTrace(BaseModel):
    """Complete trace of a single run."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    task_input: str
    task_domain: str = ""
    task_complexity: float = 0.5
    
    masks_generated: MaskWeights = None
    
    events: list[TraceEvent] = Field(default_factory=list)
    
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    
    user_approved: bool = False
    user_feedback: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def save_to_file(self, path: str):
        """Write trace as JSONL (one event per line)."""
        with open(path, "w") as f:
            for event in self.events:
                f.write(event.model_dump_json() + "\n")

class FeedbackCorrectionPair(BaseModel):
    """User feedback: rejected output + correction."""
    run_id: str  # which run this feedback is for
    rejected_output: str
    correction: str
    delta: dict = Field(default_factory=dict)
    # {
    #   "scope": "task",  # where to apply correction
    #   "aspect": "temperature",
    #   "old_value": 0.7,
    #   "new_value": 0.3,
    #   "rationale": "Too verbose; need concise planning.",
    # }
    
    confidence: float = 0.8
    created_at: datetime = Field(default_factory=datetime.utcnow)
    applied: bool = False  # did we use this to retrain mask network?

# ============================================================================
# CONFIG & SYSTEM
# ============================================================================

class IdentityConfig(BaseModel):
    """User identity configuration."""
    name: str
    role: str = "staff_engineer"
    domains: list[str] = ["ml", "systems"]
    principles_file: str  # path to P_global JSON
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Alice",
                "role": "staff_engineer",
                "domains": ["ml", "robotics"],
                "principles_file": ".minime/identity.json",
            }
        }

class MiniMeConfig(BaseModel):
    """Global MiniMe configuration."""
    identity: IdentityConfig
    memory_store: str = "sqlite"  # "sqlite" | "pgvector"
    memory_db_path: str = ".minime/memory.db"
    
    embeddings_provider: str = "openai"  # "openai" | "ollama"
    embeddings_model: str = "text-embedding-3-small"
    
    llm_provider: str = "openai"  # "openai" | "claude" | "ollama"
    llm_model: str = "gpt-4"
    llm_api_key: str = ""  # from env or secrets
    
    token_budget_per_run: int = 100_000
    approval_required_for_execute: bool = True
    
    allowlist_file: str = ".minime/allowlist.yaml"
    traces_dir: str = ".minime/traces"
    feedback_file: str = ".minime/feedback.jsonl"
    
    class Config:
        env_prefix = "MINIME_"
```

---

## 6. MASK SYSTEM IMPLEMENTATION PLAN

### The Cognitive Mask Hierarchy

```
Global Mask (P_global)
    ↓
    + Domain Mask (α weight: 0-1)
    ↓
    + Task Mask (β weight: 0-1)
    ↓
    + Agent Mask (γ weight: 0-1)
    ↓
    = Effective Thinking State (P_effective)
```

**Math (Intuition)**:
```
P_effective = (1 - α - β - γ) * P_global
            + α * P_domain
            + β * P_task
            + γ * P_agent

where α + β + γ <= 1.0 (room for default reasoning)
```

### MVP: Heuristic Mask Weights

**No learnable parameters initially.** Fixed rules determine masks.

```python
class MaskHeuristic:
    """Heuristic mask generator (no training)."""
    
    def generate(self, request: MaskRequest) -> MaskWeights:
        """
        Heuristic rules for α, β, γ based on task + domain.
        """
        task_embedding = request.task_embedding
        domain = request.domain
        complexity = request.complexity
        
        # Domain weight: boost if domain detected
        alpha = 0.6 if domain != "general" else 0.1
        
        # Task weight: boost for complex tasks
        beta = min(0.5, complexity * 0.7)
        
        # Agent weight: depends on task type
        gamma = 0.2
        
        # Retrieval k: scale by complexity
        k = 3 if complexity < 0.3 else (5 if complexity < 0.7 else 8)
        
        # Temperature: lower for structured output, higher for creative
        is_creative = any(w in task_embedding for w in ["design", "explore"])
        temperature = 0.9 if is_creative else 0.5
        
        # Rigor: strict for code, looser for brainstorming
        rigor = 1.2 if "code" in domain else 0.8
        
        # Agent routing
        agent_routing = self._route_agents(domain, task_embedding)
        
        return MaskWeights(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            retrieval_k=k,
            temperature=temperature,
            rigor=rigor,
            agent_routing=agent_routing,
        )
    
    def _route_agents(self, domain: str, task_emb: np.ndarray) -> dict:
        """Decide which agents to involve."""
        # Simplified routing
        return {
            "architect": 0.3 if "design" in domain else 0.2,
            "builder": 0.5,
            "critic": 0.3,
            "integrator": 0.1,
        }
```

### Phase 2: Trainable Mask Network

**Architecture** (post-MVP, but schema now):

```python
class MaskNetworkMLPv1(torch.nn.Module):
    """
    Small MLP that learns mask weights from user feedback.
    
    Input: [task_embedding (384), domain_embedding (384), complexity (1)]
    Hidden: [256, GELU], [128, GELU], [64, LayerNorm]
    Output: [α, β, γ, k, temperature, rigor, agent_routing] (≈20 dims)
    """
    def __init__(self, input_dim: int = 769, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output head
        layers.append(torch.nn.Linear(prev_dim, 20))  # [α, β, γ, k, temp, rigor, routing*4]
        
        self.net = torch.nn.Sequential(*layers)
        self.output_layer = layers[-1]  # for loss computation
    
    def forward(self, task_emb: torch.Tensor, domain_emb: torch.Tensor, complexity: float) -> MaskWeights:
        x = torch.cat([task_emb, domain_emb, torch.tensor([complexity])])
        logits = self.net(x.unsqueeze(0))  # (1, 20)
        
        # Postprocess logits → bounded outputs
        alpha = torch.sigmoid(logits[0, 0]).item()
        beta = torch.sigmoid(logits[0, 1]).item()
        gamma = torch.sigmoid(logits[0, 2]).item()
        k = int(torch.clamp(logits[0, 3], 3, 10).item())
        temperature = torch.clamp(logits[0, 4], 0.0, 2.0).item()
        rigor = torch.clamp(logits[0, 5], 0.5, 1.5).item()
        
        agent_routing = {
            "architect": torch.sigmoid(logits[0, 6]).item(),
            "builder": torch.sigmoid(logits[0, 7]).item(),
            "critic": torch.sigmoid(logits[0, 8]).item(),
            "integrator": torch.sigmoid(logits[0, 9]).item(),
        }
        
        return MaskWeights(
            alpha=alpha, beta=beta, gamma=gamma,
            retrieval_k=k, temperature=temperature, rigor=rigor,
            agent_routing=agent_routing,
        )
```

**Training Loop**:
```python
def train_mask_network(
    model: MaskNetworkMLPv1,
    feedback_pairs: list[FeedbackCorrectionPair],
    epochs: int = 5,
    lr: float = 1e-3,
):
    """Train mask network on user corrections."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for pair in feedback_pairs:
            # Get original task embedding
            task_emb = embed_text(pair.run_id)  # from trace
            
            # Target: what mask weights would have produced correct output
            target_delta = pair.delta
            target_weights = compute_target_mask_weights(target_delta)
            
            # Predict
            pred_weights = model(task_emb, domain_emb, complexity)
            
            # Loss: MSE on important fields
            loss = loss_fn(
                torch.tensor([pred_weights.alpha, pred_weights.beta, ...]),
                torch.tensor([target_weights.alpha, target_weights.beta, ...])
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: loss={total_loss/len(feedback_pairs):.4f}")
    
    # Save model
    torch.save(model.state_dict(), ".minime/models/mask_network.pth")
```

### Temporary Context Abstraction (Subspace Projection)

**Concept**: When a task only involves a few domains, spawn a temporary lower-dimensional subspace and reason there.

**Implementation** (Phase 2):

```python
class TemporarySubspace:
    """Spawns a task-specific subspace for reasoning."""
    
    def __init__(self, relevant_principles: list[IdentityPrinciple], dims: int = 128):
        """Initialize subspace with relevant principles only."""
        self.relevant_principles = relevant_principles
        self.dims = dims
        
        # Compute projection matrix: PCA of relevant principle vectors
        vectors = np.array([p.vector for p in relevant_principles])
        self.pca = PCA(n_components=dims)
        self.projection_matrix = self.pca.fit_transform(vectors)
    
    def project_vector(self, vector: np.ndarray) -> np.ndarray:
        """Project external vector into this subspace."""
        return self.pca.transform(vector.reshape(1, -1))[0]
    
    def renormalize(self):
        """Rescale principle magnitudes within subspace to sum to 1."""
        total_magnitude = sum(p.magnitude for p in self.relevant_principles)
        for p in self.relevant_principles:
            p.magnitude /= total_magnitude
    
    def merge_or_discard(self, keep: bool = False):
        """
        Option 1 (keep=True): Merge changes back to global identity.
                (weighted by confidence)
        Option 2 (keep=False): Discard changes, revert to global.
        """
        if keep:
            for p in self.relevant_principles:
                # Apply weighted update to global identity
                # (details omitted for brevity)
                pass
        # Subspace is then garbage-collected
```

---

## 7. MULTI-AGENT ORCHESTRATION PLAN

### Agent Definitions (What Each Agent Sees + Does)

**Architect Agent**
- **Sees**: Task, relevant design principles, past architecture decisions
- **Does**: Generates system structure (modules, dependencies, file layout)
- **Input**: Task description + domain
- **Output**: `{modules: [...], dependencies: {...}, file_structure: {...}}`
- **Mask**: High weight on domain principles (α ≈ 0.7)
- **Example Prompt Block**:
  ```
  You are the Architect. Your job is to design the system structure.
  
  IDENTITY (weighted α=0.7):
  - Modularity-first: each module has a single responsibility
  - Reversible: no breaking changes within a module
  - Abstraction-first: define interfaces before implementations
  
  DOMAIN KNOWLEDGE:
  [Retrieved notes on system design patterns]
  
  TASK:
  Plan a REST API for user authentication.
  
  OUTPUT: Return JSON with:
  {
    "modules": ["auth", "db", "cache", ...],
    "dependencies": {"auth": ["db"]},
    "file_structure": {"src/auth/": ["handler.py", "schema.py", ...]},
  }
  ```

**Builder Agent**
- **Sees**: Architecture, relevant code examples, style guides
- **Does**: Implements modules (write files, apply diffs)
- **Input**: Architect output + detailed implementation task
- **Output**: Code (with proposed write/apply actions)
- **Mask**: Balanced α/β (α=0.4, β=0.5)
- **Actions**: write_file, apply_patch
- **Safety**: Code reviewed by Critic before execution

**Critic Agent**
- **Sees**: Code output, identity principles, constraints
- **Does**: Adversarial review; flags violations
- **Input**: Code + reasoning
- **Output**: `{valid: bool, violations: [...], suggestions: [...]}`
- **Mask**: High rigor (rigor=1.3), low temperature (0.2)
- **Decision**: If violations found, loop back to Builder (max 3 times)
- **Example Prompt**:
  ```
  You are the Critic. Your job is to validate output.
  
  CONSTRAINTS:
  - No hardcoded secrets
  - All functions documented
  - Type hints on public APIs
  - <2000 tokens per function
  
  CODE UNDER REVIEW:
  [Builder's code]
  
  OUTPUT: Return JSON with:
  {
    "valid": false,
    "violations": [
      "Function `authenticate()` missing type hints on params",
      "Hardcoded API_KEY in line 42"
    ],
    "suggestions": [
      "Add `: str -> bool` signature",
      "Move API_KEY to environment variable"
    ]
  }
  ```

**Integrator Agent**
- **Sees**: Outputs from multiple agents (Architect, Builder, Critic)
- **Does**: Merges/reconciles conflicting outputs
- **Input**: List of agent outputs + decisions
- **Output**: Final unified output (or summary of conflicts)
- **Mask**: Moderate weights (α=0.3, β=0.3, γ=0.4)

### Agent Orchestration Flow

```python
async def orchestrate_agents(
    task: str,
    mask_weights: MaskWeights,
    max_iterations: int = 3,
) -> AgentOutput:
    """
    Run agents in sequence/parallel with iteration control.
    """
    
    # Step 1: Architect designs
    architect_output = await architect_agent.run(
        task=task,
        mask=mask_weights,
        context=context_broker.assemble(...),
    )
    
    # Step 2: Builder implements (based on architecture)
    builder_output = await builder_agent.run(
        task=task,
        architecture=architect_output,
        mask=mask_weights,
    )
    
    # Step 3: Critic reviews (loop until valid or max_iterations)
    iteration = 0
    while iteration < max_iterations:
        critic_output = await critic_agent.review(
            code=builder_output,
            constraints=identity.principles,
            mask=mask_weights,
        )
        
        if critic_output.valid:
            break
        
        # Request revision
        builder_output = await builder_agent.revise(
            code=builder_output,
            feedback=critic_output.violations,
            suggestions=critic_output.suggestions,
        )
        iteration += 1
    
    if not critic_output.valid:
        raise ValueError(f"Code failed critic after {max_iterations} attempts")
    
    # Step 4: Integrate (if multiple agents contributed)
    final_output = await integrator_agent.integrate(
        architect=architect_output,
        builder=builder_output,
        critic=critic_output,
    )
    
    return final_output
```

### Critic Constraint Enforcement

**Constraint Categories**:

```python
class ConstraintSet(BaseModel):
    """Set of constraints Critic enforces."""
    
    # Code style
    max_function_lines: int = 50
    require_type_hints: bool = True
    require_docstrings: bool = True
    
    # Safety
    disallow_hardcoded_secrets: bool = True
    disallow_shell_injection: list[str] = ["eval", "exec", "shell=True"]
    
    # Performance
    max_cyclomatic_complexity: int = 5
    
    # Architectural
    modules_must_have_interface: bool = True
    no_circular_imports: bool = True
    
    def check(self, code: str) -> tuple[bool, list[str]]:
        """Analyze code against constraints."""
        violations = []
        
        # Check 1: function length
        for func in parse_functions(code):
            if len(func.lines) > self.max_function_lines:
                violations.append(
                    f"Function `{func.name}` is {len(func.lines)} lines "
                    f"(max {self.max_function_lines})"
                )
        
        # Check 2: hardcoded secrets
        if self.disallow_hardcoded_secrets:
            for match in re.finditer(r'(password|api_key|secret)\s*=\s*["\']', code):
                violations.append(f"Hardcoded secret at line {match.lineno}")
        
        # ... more checks
        
        return len(violations) == 0, violations
```

### Iteration + Termination

```python
class IterationControl:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.iteration = 0
    
    def should_continue(self, critic_feedback: CriticOutput) -> bool:
        """Decide whether to continue or accept output."""
        if critic_feedback.valid:
            return False  # Stop, output is valid
        
        if self.iteration >= self.max_iterations:
            return False  # Stop, max attempts reached
        
        # Continue: builder revises
        self.iteration += 1
        return True
```

---

## 8. REPOSITORY SCAFFOLD (EXACT TREE)

```
minime/
├── .gitignore
├── README.md
├── pyproject.toml              # Poetry or setuptools config
├── requirements.txt            # Pinned deps
├── Makefile                    # Common tasks
│
├── minime/
│   ├── __init__.py
│   ├── main.py                 # CLI entrypoint (Typer)
│   ├── config.py               # Config loading + validation
│   ├── constants.py            # Magic numbers, paths
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── identity.py         # P_global, IdentityPrinciple
│   │   ├── memory.py           # MemoryChunk, MemoryStore interface
│   │   ├── mask_network.py     # MaskHeuristic, MaskWeights
│   │   ├── context_broker.py   # Context assembly + budget
│   │   ├── orchestrator.py     # Agent loop + flow control
│   │   └── tracing.py          # RunTrace, TraceEvent logging
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py             # Agent ABC
│   │   ├── architect.py        # Architect agent
│   │   ├── builder.py          # Builder agent
│   │   ├── critic.py           # Critic agent
│   │   ├── integrator.py       # Integrator agent
│   │   └── llm_prompts.py      # Shared prompt templates
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py             # LLMProvider, EmbeddingsProvider ABCs
│   │   ├── openai_provider.py  # OpenAI implementation
│   │   ├── ollama_provider.py  # Ollama (local) fallback
│   │   └── mock_provider.py    # For testing
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py             # VectorStore ABC
│   │   ├── sqlite_store.py     # Local SQLite + numpy
│   │   ├── pgvector_store.py   # PostgreSQL (future)
│   │   └── migrations.py       # Schema versioning
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py             # Tool ABC
│   │   ├── definitions.py      # Tool schema definitions
│   │   ├── executor.py         # Tool execution + allowlist
│   │   └── allowlist.py        # Allowlist enforcement
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py          # All Pydantic models
│   │   └── types.py            # Type aliases
│   │
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py       # Embedding helpers
│       ├── chunking.py         # Text chunking
│       ├── logger.py           # Logging setup
│       └── validators.py       # Input validation
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── test_identity.py
│   ├── test_memory.py
│   ├── test_mask_network.py
│   ├── test_context_broker.py
│   ├── test_agents.py
│   ├── test_tools.py
│   ├── test_orchestrator.py    # E2E
│   └── test_integration.py     # Full flow with mocks
│
├── scripts/
│   ├── init_project.sh         # minime init implementation
│   ├── train_mask.py           # Offline mask training (Phase 2)
│   ├── import_notes.py         # Batch import notes → memory
│   ├── inspect_traces.py       # Load + pretty-print traces
│   └── benchmark.py            # Performance profiling
│
├── .minime/ (generated at runtime)
│   ├── config.yaml             # Main config
│   ├── identity.json           # P_global serialized
│   ├── allowlist.yaml          # Tool allowlist
│   ├── memory.db               # SQLite vector store
│   ├── memory/
│   │   └── chunks/             # Cached chunk metadata
│   ├── models/
│   │   └── mask_network.pth    # Trained mask network (Phase 2)
│   ├── traces/
│   │   ├── {run_id_1}.jsonl    # One trace per run
│   │   └── {run_id_2}.jsonl
│   └── feedback.jsonl          # User corrections + deltas
│
├── docs/
│   ├── ARCHITECTURE.md         # Full design doc
│   ├── API.md                  # API reference
│   ├── QUICKSTART.md
│   ├── DEVELOPMENT.md
│   └── EXAMPLES.md             # Use cases
│
└── .github/
    └── workflows/
        ├── test.yaml           # CI: pytest
        └── lint.yaml           # CI: ruff + mypy
```

**Files per Layer** (at a glance):
- Identity: `minime/core/identity.py` (150 lines)
- Memory: `minime/core/memory.py`, `minime/storage/sqlite_store.py` (300 lines total)
- Mask: `minime/core/mask_network.py` (250 lines)
- Context: `minime/core/context_broker.py` (200 lines)
- Agents: `minime/agents/*.py` (500 lines total)
- Orchestrator: `minime/core/orchestrator.py` (300 lines)
- Tools: `minime/tools/executor.py`, `minime/tools/allowlist.py` (200 lines)
- Tracing: `minime/core/tracing.py` (100 lines)
- CLI: `minime/main.py` (400 lines)

**Estimated MVP Total: ~2500 lines of code**

---

## 9. STARTER CODE (COPY-PASTE READY)

### 9.1 CLI Entrypoint (`minime/main.py`)

```python
"""MiniMe CLI entrypoint using Typer."""
import asyncio
import json
from pathlib import Path
from typing import Optional
import typer

from minime.config import MiniMeConfig, load_config
from minime.core.orchestrator import MiniMeOrchestrator
from minime.models.schemas import RunTrace
from minime.utils.logger import setup_logger

app = typer.Typer(help="MiniMe: Your personal cognitive engine")
logger = setup_logger(__name__)


@app.command()
def init(
    name: str = typer.Option("Anonymous", "--name", "-n"),
    config_dir: str = typer.Option(".minime", "--config-dir"),
):
    """Initialize MiniMe in current directory."""
    config_path = Path(config_dir)
    config_path.mkdir(exist_ok=True)
    
    # Create default identity
    identity_json = {
        "name": name,
        "role": "engineer",
        "domains": ["ml", "systems"],
        "principles": [
            {
                "name": "modularity",
                "description": "Single responsibility per module",
                "magnitude": 1.0,
                "scope": "global",
            },
        ],
    }
    (config_path / "identity.json").write_text(json.dumps(identity_json, indent=2))
    
    # Create default config
    config = {
        "identity": {"name": name, "principles_file": str(config_path / "identity.json")},
        "memory_store": "sqlite",
        "memory_db_path": str(config_path / "memory.db"),
        "llm_provider": "openai",
        "llm_model": "gpt-4",
        "token_budget_per_run": 100_000,
        "approval_required_for_execute": True,
        "allowlist_file": str(config_path / "allowlist.yaml"),
        "traces_dir": str(config_path / "traces"),
        "feedback_file": str(config_path / "feedback.jsonl"),
    }
    (config_path / "config.yaml").write_text(__import__("yaml").dump(config))
    
    # Create default allowlist
    allowlist = {
        "write": {
            "patterns": ["src/**/*.py", "config/**/*.yaml", "docs/**/*.md"],
            "exclude": [".env", "*.key"],
        },
        "run": {
            "trusted": ["python -m pytest", "git status"],
            "untrusted": ["rm -rf", "curl"],
        },
    }
    (config_path / "allowlist.yaml").write_text(__import__("yaml").dump(allowlist))
    
    (config_path / "traces").mkdir(exist_ok=True)
    
    typer.echo(f"✓ Initialized MiniMe at {config_dir}")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question or task"),
    config_dir: str = typer.Option(".minime", "--config-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't call LLM, show context"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Ask MiniMe to help with a task."""
    try:
        config = load_config(config_dir)
        if verbose:
            logger.setLevel("DEBUG")
        
        orchestrator = MiniMeOrchestrator(config)
        
        # Run async orchestration
        trace = asyncio.run(orchestrator.run(
            task=query,
            dry_run=dry_run,
            verbose=verbose,
        ))
        
        # Display result
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        
        if trace.events:
            last_event = trace.events[-1]
            if last_event.event_type == "action_proposal":
                actions = last_event.data.get("actions", [])
                print(f"\nProposed {len(actions)} actions:")
                for i, action in enumerate(actions, 1):
                    print(f"  {i}. {action['tool']} {action['args']}")
                
                if not dry_run and config.approval_required_for_execute:
                    if typer.confirm("Proceed with execution?"):
                        # Execute (details in ExecutionResult)
                        asyncio.run(orchestrator.execute_approved_actions(actions))
                        typer.echo("✓ Executed successfully")
                    else:
                        typer.echo("Cancelled")
        
        # Save trace
        trace_file = Path(config.traces_dir) / f"{trace.id}.jsonl"
        trace.save_to_file(str(trace_file))
        typer.echo(f"\n✓ Trace saved to {trace_file}")
    
    except Exception as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def import_notes(
    path: str = typer.Argument(..., help="Path to notes or directory"),
    domain: str = typer.Option("general", "--domain", "-d"),
    config_dir: str = typer.Option(".minime", "--config-dir"),
):
    """Import notes into memory."""
    config = load_config(config_dir)
    from minime.core.memory import MemoryStore
    from minime.storage.sqlite_store import SQLiteStore
    
    memory_store = SQLiteStore(config.memory_db_path)
    
    # Load files
    import_path = Path(path)
    if import_path.is_dir():
        files = list(import_path.glob("**/*.md")) + list(import_path.glob("**/*.txt"))
    else:
        files = [import_path]
    
    typer.echo(f"Importing {len(files)} files...")
    
    for file_path in files:
        text = file_path.read_text()
        asyncio.run(memory_store.store_chunk(
            text=text,
            metadata={"source": str(file_path), "domain": domain},
        ))
    
    typer.echo(f"✓ Imported {len(files)} files")


@app.command()
def trace(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    config_dir: str = typer.Option(".minime", "--config-dir"),
):
    """Pretty-print a trace file."""
    config = load_config(config_dir)
    trace_file = Path(config.traces_dir) / f"{run_id}.jsonl"
    
    if not trace_file.exists():
        typer.echo(f"✗ Trace {run_id} not found", err=True)
        return
    
    # Read + pretty-print
    from minime.utils.logger import format_trace
    print(format_trace(trace_file))


@app.command()
def config(
    action: str = typer.Argument("view", help="view | edit | reset"),
    config_dir: str = typer.Option(".minime", "--config-dir"),
):
    """Manage configuration."""
    config_file = Path(config_dir) / "config.yaml"
    
    if action == "view":
        print(config_file.read_text())
    elif action == "edit":
        import subprocess
        editor = subprocess.os.environ.get("EDITOR", "vim")
        subprocess.call([editor, str(config_file)])
    elif action == "reset":
        if typer.confirm("Reset to defaults?"):
            init(config_dir=config_dir)


if __name__ == "__main__":
    app()
```

### 9.2 Provider Interface + OpenAI Implementation

```python
# minime/providers/base.py
"""Abstract provider interfaces."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class CompletionResponse:
    text: Optional[str] = None
    tool_calls: Optional[list] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[list] = None,
        response_format: str = "text",
    ) -> CompletionResponse:
        """Generate response from LLM."""
        pass


class EmbeddingsProvider(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed single text; returns 1536-dim vector."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        pass


# minime/providers/openai_provider.py
"""OpenAI API provider."""
import json
from typing import Optional
from openai import AsyncOpenAI
from minime.providers.base import LLMProvider, EmbeddingsProvider, CompletionResponse
import os


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Pricing (as of Dec 2024)
        self.pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        }
    
    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[list] = None,
        response_format: str = "text",
    ) -> CompletionResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Handle tools
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                }
                for tool in tools
            ]
        
        # Handle response format
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**kwargs)
        
        # Parse response
        text = None
        tool_calls = None
        
        for choice in response.choices:
            if choice.message.content:
                text = choice.message.content
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                    for tc in choice.message.tool_calls
                ]
        
        # Calculate cost
        pricing = self.pricing.get(self.model, self.pricing["gpt-3.5-turbo"])
        cost = (
            response.usage.prompt_tokens * pricing["input"] +
            response.usage.completion_tokens * pricing["output"]
        )
        
        return CompletionResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=cost,
        )


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def embed_text(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]
```

### 9.3 Context Broker (Block Assembly + Budget)

```python
# minime/core/context_broker.py
"""Context assembly with token budgeting."""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContextBlock:
    name: str
    content: str
    weight: float = 1.0  # how important is this block
    token_estimate: int = 0


class ContextBroker:
    """Assembles prompt blocks with token budget enforcement."""
    
    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget
        self.blocks: list[ContextBlock] = []
    
    def add_block(self, name: str, content: str, weight: float = 1.0):
        """Add a context block."""
        token_estimate = len(content.split()) * 1.3  # rough estimate
        block = ContextBlock(name, content, weight, int(token_estimate))
        self.blocks.append(block)
    
    def assemble(self) -> tuple[str, dict]:
        """
        Assemble blocks into prompt, respecting budget.
        Returns (assembled_prompt, metadata)
        """
        # Sort blocks by weight (highest first)
        sorted_blocks = sorted(self.blocks, key=lambda b: b.weight, reverse=True)
        
        prompt_parts = []
        tokens_used = 0
        blocks_used = []
        blocks_dropped = []
        
        for block in sorted_blocks:
            if tokens_used + block.token_estimate <= self.token_budget:
                prompt_parts.append(f"## {block.name}\n{block.content}\n")
                tokens_used += block.token_estimate
                blocks_used.append(block.name)
            else:
                blocks_dropped.append(block.name)
        
        if blocks_dropped:
            prompt_parts.append(f"\n[Dropped due to budget: {', '.join(blocks_dropped)}]\n")
        
        assembled = "\n".join(prompt_parts)
        
        return assembled, {
            "blocks_used": blocks_used,
            "blocks_dropped": blocks_dropped,
            "tokens_used": tokens_used,
            "budget": self.token_budget,
        }
```

### 9.4 Mask System (Heuristic Version)

```python
# minime/core/mask_network.py
"""Mask system for shaping reasoning."""
import numpy as np
from minime.models.schemas import MaskWeights, MaskRequest


class MaskHeuristic:
    """Heuristic mask generator (no learning, MVP only)."""
    
    def generate(self, request: MaskRequest) -> MaskWeights:
        """Generate mask weights based on task."""
        domain = request.domain.lower()
        complexity = request.complexity
        
        # Heuristic rules
        alpha = 0.6 if domain != "general" else 0.1
        beta = min(0.5, complexity * 0.7)
        gamma = 0.2
        
        k = 3 if complexity < 0.3 else (5 if complexity < 0.7 else 8)
        temperature = 0.5 if "code" in domain else 0.8
        rigor = 1.2 if "code" in domain else 0.9
        
        return MaskWeights(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            retrieval_k=k,
            temperature=temperature,
            rigor=rigor,
            agent_routing={
                "architect": 0.3 if "design" in domain else 0.2,
                "builder": 0.5,
                "critic": 0.3,
                "integrator": 0.1,
            },
        )
```

### 9.5 Vector Memory (SQLite + NumPy)

```python
# minime/storage/sqlite_store.py
"""Local SQLite vector store."""
import sqlite3
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from minime.models.schemas import MemoryChunk, RetrievalResult


class SQLiteStore:
    """Local vector store using SQLite + numpy arrays."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create schema if needed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata
                ON chunks (metadata)
            """)
            conn.commit()
    
    async def store_chunk(self, text: str, metadata: dict):
        """Store a chunk with embedding."""
        from minime.providers.openai_provider import OpenAIEmbeddingsProvider
        embeddings = OpenAIEmbeddingsProvider()
        
        embedding = await embeddings.embed_text(text)
        embedding_blob = base64.b64encode(np.array(embedding).tobytes())
        
        chunk_id = str(__import__("uuid").uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO chunks (id, text, embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk_id,
                text,
                embedding_blob,
                json.dumps(metadata),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
        
        return chunk_id
    
    async def retrieve(self, query: str, k: int = 5, filters: dict = None) -> RetrievalResult:
        """Retrieve top-k similar chunks."""
        from minime.providers.openai_provider import OpenAIEmbeddingsProvider
        embeddings = OpenAIEmbeddingsProvider()
        
        query_embedding = await embeddings.embed_text(query)
        query_vector = np.array(query_embedding)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, text, embedding, metadata FROM chunks
            """)
            
            results = []
            for chunk_id, text, embedding_blob, metadata_json in cursor:
                embedding = np.frombuffer(base64.b64decode(embedding_blob), dtype=np.float32)
                
                # Cosine similarity
                similarity = np.dot(query_vector, embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(embedding) + 1e-8
                )
                
                results.append((
                    MemoryChunk(
                        id=chunk_id,
                        text=text,
                        metadata=json.loads(metadata_json),
                    ),
                    similarity,
                ))
            
            # Sort by similarity and return top-k
            results.sort(key=lambda x: x[1], reverse=True)
            chunks = [r[0] for r in results[:k]]
            scores = [r[1] for r in results[:k]]
            
            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query_tokens=len(query.split()),
                retrieval_weights_applied={},
            )
```

### 9.6 Tool Executor + Allowlist

```python
# minime/tools/executor.py
"""Execute tools with allowlist enforcement."""
import subprocess
from pathlib import Path
from typing import Optional
import yaml

from minime.models.schemas import ExecutionResult


class ToolExecutor:
    """Execute tools safely."""
    
    def __init__(self, allowlist_file: str):
        with open(allowlist_file) as f:
            self.allowlist = yaml.safe_load(f)
    
    async def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write file to disk."""
        path = Path(path)
        
        # Check allowlist
        if not self._is_allowed_write(str(path)):
            return ExecutionResult(
                success=False,
                error=f"Path {path} not in allowlist",
            )
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return ExecutionResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    async def run_command(
        self,
        cmd: str,
        cwd: str = ".",
        allowlist_tag: str = "trusted",
    ) -> ExecutionResult:
        """Run shell command."""
        if not self._is_allowed_run(cmd, allowlist_tag):
            return ExecutionResult(
                success=False,
                error=f"Command not in {allowlist_tag} allowlist",
            )
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                timeout=30,
                text=True,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, error="Command timed out (30s)")
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    def _is_allowed_write(self, path: str) -> bool:
        import fnmatch
        patterns = self.allowlist.get("write", {}).get("patterns", [])
        excludes = self.allowlist.get("write", {}).get("exclude", [])
        
        allowed = any(fnmatch.fnmatch(path, p) for p in patterns)
        denied = any(fnmatch.fnmatch(path, e) for e in excludes)
        
        return allowed and not denied
    
    def _is_allowed_run(self, cmd: str, tag: str) -> bool:
        allowed_cmds = self.allowlist.get("run", {}).get(tag, [])
        return any(cmd.startswith(allowed) for allowed in allowed_cmds)
```

### 9.7 Tracing

```python
# minime/core/tracing.py
"""Trace logging for debugging and learning."""
import json
from pathlib import Path
from datetime import datetime
from minime.models.schemas import RunTrace, TraceEvent


class TraceWriter:
    """Write events to JSONL trace file."""
    
    def __init__(self, trace: RunTrace):
        self.trace = trace
    
    def add_event(
        self,
        event_type: str,
        data: dict,
        level: str = "info",
    ):
        """Add event to trace."""
        event = TraceEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            level=level,
            data=data,
        )
        self.trace.events.append(event)
    
    def save(self, path: str):
        """Save trace as JSONL."""
        with open(path, "w") as f:
            for event in self.trace.events:
                f.write(event.model_dump_json() + "\n")
```

### 9.8 Orchestrator (Main Flow)

```python
# minime/core/orchestrator.py
"""Main orchestration logic."""
import asyncio
from minime.config import MiniMeConfig
from minime.core.mask_network import MaskHeuristic
from minime.core.context_broker import ContextBroker
from minime.core.tracing import TraceWriter
from minime.models.schemas import RunTrace, MaskRequest
from minime.providers.openai_provider import OpenAIProvider, OpenAIEmbeddingsProvider
from minime.storage.sqlite_store import SQLiteStore


class MiniMeOrchestrator:
    """Main orchestration engine."""
    
    def __init__(self, config: MiniMeConfig):
        self.config = config
        self.llm = OpenAIProvider(model=config.llm_model)
        self.embeddings = OpenAIEmbeddingsProvider()
        self.memory = SQLiteStore(config.memory_db_path)
        self.mask_engine = MaskHeuristic()
    
    async def run(
        self,
        task: str,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> RunTrace:
        """Execute a task end-to-end."""
        trace = RunTrace(task_input=task)
        writer = TraceWriter(trace)
        
        # Step 1: Embed task + detect domain
        task_embedding = await self.embeddings.embed_text(task)
        domain = self._detect_domain(task)
        complexity = self._estimate_complexity(task)
        
        trace.task_domain = domain
        trace.task_complexity = complexity
        
        writer.add_event("task_start", {
            "task": task,
            "domain": domain,
            "complexity": complexity,
        })
        
        # Step 2: Generate masks
        mask_request = MaskRequest(
            task_embedding=task_embedding,
            domain=domain,
            complexity=complexity,
        )
        masks = self.mask_engine.generate(mask_request)
        trace.masks_generated = masks
        
        writer.add_event("mask_generation", {
            "alpha": masks.alpha,
            "beta": masks.beta,
            "gamma": masks.gamma,
            "retrieval_k": masks.retrieval_k,
            "temperature": masks.temperature,
        })
        
        # Step 3: Retrieve context
        memory_result = await self.memory.retrieve(
            query=task,
            k=masks.retrieval_k,
        )
        
        writer.add_event("retrieval", {
            "query": task,
            "k": masks.retrieval_k,
            "results_count": len(memory_result.chunks),
            "scores": memory_result.scores,
        })
        
        # Step 4: Assemble context
        context_broker = ContextBroker(token_budget=self.config.token_budget_per_run)
        
        # Identity block
        identity_content = f"You are a {self.config.identity.role}. Domains: {', '.join(self.config.identity.domains)}"
        context_broker.add_block("Identity", identity_content, weight=masks.alpha)
        
        # Task block
        context_broker.add_block("Task", task, weight=1.0)
        
        # Memory block
        memory_content = "\n---\n".join([
            f"[{score:.2f}] {chunk.text[:200]}"
            for chunk, score in zip(memory_result.chunks, memory_result.scores)
        ])
        context_broker.add_block("Retrieved Knowledge", memory_content, weight=masks.beta)
        
        assembled, metadata = context_broker.assemble()
        
        writer.add_event("context_assembly", {
            "blocks_used": metadata["blocks_used"],
            "blocks_dropped": metadata["blocks_dropped"],
            "tokens_used": metadata["tokens_used"],
        })
        
        # Step 5: Call LLM (if not dry_run)
        if not dry_run:
            completion = await self.llm.complete(
                prompt=assembled,
                system="You are a thoughtful engineer. Provide structured output.",
                temperature=masks.temperature,
                response_format="json_object",
            )
            
            trace.total_tokens_used = completion.prompt_tokens + completion.completion_tokens
            trace.total_cost_usd = completion.cost_usd
            
            writer.add_event("llm_call", {
                "model": self.config.llm_model,
                "tokens": completion.prompt_tokens + completion.completion_tokens,
                "cost_usd": completion.cost_usd,
            })
            
            output = completion.text or ""
        else:
            output = "[DRY RUN: No LLM call made]"
        
        # Step 6: Propose actions
        actions = [
            {
                "tool": "write_file",
                "args": {"path": "example.py", "content": "# Example"},
            }
        ]
        
        writer.add_event("action_proposal", {
            "actions": actions,
            "reasoning": output,
        })
        
        return trace
    
    def _detect_domain(self, task: str) -> str:
        """Simple domain detection."""
        keywords = {
            "ml": ["model", "train", "data", "neural", "gradient"],
            "web": ["api", "rest", "http", "server", "deploy"],
            "robotics": ["robot", "motion", "control", "sim"],
        }
        
        task_lower = task.lower()
        for domain, words in keywords.items():
            if any(word in task_lower for word in words):
                return domain
        return "general"
    
    def _estimate_complexity(self, task: str) -> float:
        """Estimate task complexity (0-1)."""
        length = len(task.split())
        complexity = min(1.0, length / 50)  # Rough heuristic
        return complexity
    
    async def execute_approved_actions(self, actions: list[dict]):
        """Execute approved actions."""
        from minime.tools.executor import ToolExecutor
        executor = ToolExecutor(self.config.allowlist_file)
        
        for action in actions:
            tool = action["tool"]
            args = action["args"]
            
            if tool == "write_file":
                result = await executor.write_file(**args)
            elif tool == "run_command":
                result = await executor.run_command(**args)
            else:
                continue
            
            print(f"[{tool}] {result.output}")
```

### 9.9 Config Loading

```python
# minime/config.py
"""Config loading and validation."""
import os
from pathlib import Path
import yaml
from minime.models.schemas import MiniMeConfig


def load_config(config_dir: str = ".minime") -> MiniMeConfig:
    """Load config from file + env overrides."""
    config_path = Path(config_dir) / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Apply env overrides
    config_dict["llm_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if os.getenv("MINIME_LLM_PROVIDER"):
        config_dict["llm_provider"] = os.getenv("MINIME_LLM_PROVIDER")
    if os.getenv("MINIME_LLM_MODEL"):
        config_dict["llm_model"] = os.getenv("MINIME_LLM_MODEL")
    
    return MiniMeConfig(**config_dict)
```

---

## 10. TESTING STRATEGY

### Unit Tests

```python
# tests/test_context_broker.py
import pytest
from minime.core.context_broker import ContextBroker


def test_context_assembly_respects_budget():
    """Context broker should drop blocks if over budget."""
    broker = ContextBroker(token_budget=100)
    
    broker.add_block("Identity", "x" * 50, weight=1.0)
    broker.add_block("Task", "y" * 40, weight=1.0)
    broker.add_block("Memory", "z" * 1000, weight=0.5)  # Should be dropped
    
    assembled, metadata = broker.assemble()
    
    assert "Identity" in metadata["blocks_used"]
    assert "Task" in metadata["blocks_used"]
    assert "Memory" in metadata["blocks_dropped"]
    assert metadata["tokens_used"] <= 100


def test_block_weighting():
    """Higher-weight blocks should be prioritized."""
    broker = ContextBroker(token_budget=100)
    
    broker.add_block("Low", "a" * 50, weight=0.1)
    broker.add_block("High", "b" * 60, weight=1.0)
    
    assembled, metadata = broker.assemble()
    
    assert "High" in metadata["blocks_used"]
    assert "Low" in metadata["blocks_dropped"]


# tests/test_tools.py
def test_allowlist_write_patterns():
    """Allowlist should enforce write patterns."""
    executor = ToolExecutor(".minime/allowlist.yaml")
    
    assert executor._is_allowed_write("src/main.py") == True
    assert executor._is_allowed_write(".env") == False
    assert executor._is_allowed_write("/etc/passwd") == False


def test_allowlist_command_execution():
    """Allowlist should enforce command execution."""
    executor = ToolExecutor(".minime/allowlist.yaml")
    
    assert executor._is_allowed_run("python -m pytest", "trusted") == True
    assert executor._is_allowed_run("rm -rf /", "untrusted") == False
```

### E2E Test (Mocked LLM)

```python
# tests/test_orchestrator_e2e.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from minime.core.orchestrator import MiniMeOrchestrator
from minime.config import MiniMeConfig


@pytest.fixture
def mock_config():
    return MiniMeConfig(
        identity={"name": "Test", "role": "engineer"},
        llm_model="gpt-4",
        token_budget_per_run=100_000,
        memory_db_path=":memory:",  # SQLite in-memory
    )


@pytest.mark.asyncio
async def test_orchestrator_e2e_dry_run(mock_config):
    """E2E test: task → orchestrator → trace (dry run, no LLM)."""
    orchestrator = MiniMeOrchestrator(mock_config)
    
    trace = await orchestrator.run(
        task="Plan a REST API",
        dry_run=True,
        verbose=False,
    )
    
    # Assertions
    assert trace.task_input == "Plan a REST API"
    assert trace.task_domain in ["web", "general"]
    assert len(trace.events) > 0
    assert any(e.event_type == "task_start" for e in trace.events)
    assert any(e.event_type == "context_assembly" for e in trace.events)


@pytest.mark.asyncio
async def test_orchestrator_with_mocked_llm(mock_config):
    """E2E test with mocked LLM call."""
    orchestrator = MiniMeOrchestrator(mock_config)
    
    # Mock LLM response
    mock_response = AsyncMock()
    mock_response.text = '{"plan": "REST API structure"}'
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50
    mock_response.cost_usd = 0.01
    
    with patch.object(orchestrator.llm, "complete", return_value=mock_response):
        trace = await orchestrator.run(
            task="Plan a REST API",
            dry_run=False,
        )
    
    assert trace.total_tokens_used == 150
    assert trace.total_cost_usd == 0.01
    assert any(e.event_type == "llm_call" for e in trace.events)
```

---

## 11. NEXT 10 UPGRADES (Ranked by ROI)

| Rank | Upgrade | ROI | Effort | Impact | Timeline |
|------|---------|-----|--------|--------|----------|
| **1** | **Trainable Mask Network (MLP)** | Very High | Medium | Personalization improves over time; learns from user feedback | Week 4-5 |
| **2** | **VS Code Extension (Diff-based)** | High | High | Integrates into workflow; real-time editing | Week 6-8 |
| **3** | **Advanced Retrieval (Reranking)** | High | Medium | ColBERT or BGE reranker; improves memory relevance | Week 5 |
| **4** | **Per-Project Profiles** | Medium | Low | Multiple personas; different identity per project | Week 3.5 |
| **5** | **Structured Output Strictness** | Medium | Low | Enforce JSON schema compliance; fallback repair | Week 4 |
| **6** | **Evaluation Harness** | Medium | Medium | Measure quality; run A/B tests on mask weights | Week 6 |
| **7** | **Command Streaming** | Medium | Medium | Stream LLM responses; real-time output | Week 6 |
| **8** | **PostgreSQL Migration** | Low-Medium | Medium | Production readiness; scales to many users | Week 8+ |
| **9** | **Sandboxed Execution** | High | High | Safely run generated code in containers | Week 10+ |
| **10** | **LoRA Fine-Tune Adapters** | Low | High | Optional; extreme personalization (but complexity) | Phase 3 |

### Detailed Upgrade Plans

**Upgrade #1: Trainable Mask Network (MLP)**
- Implement PyTorch MLP (2-3 hidden layers, GELU, LayerNorm)
- Training loop on feedback pairs (Section 6)
- Model checkpointing + versioning
- A/B test new weights against heuristic
- **Files**: `scripts/train_mask.py`, `minime/core/mask_network.py` (extend)

**Upgrade #2: VS Code Extension**
- Scaffold: `minime-vscode/` directory
- Commands: `/minime ask`, `/minime diff`, `/minime approve`
- Webview for trace inspection
- Diff-based apply (unified diff → VS Code diff)
- **Files**: TypeScript + React for webview

**Upgrade #3: Advanced Retrieval (Reranking)**
- Add BGE reranker model (HuggingFace)
- Rerank top-10 by relevance before top-5 selection
- Domain-specific filtering (confidence-based)
- Recency weighting (time-decay on chunk creation date)
- **Files**: `minime/retrieval/reranker.py`

**Upgrade #4: Per-Project Profiles**
- Multiple identity matrices (project-specific P_global)
- Config: `{project_name}.minime/config.yaml`
- CLI: `minime ask --project robotics`
- Shared memory pool (cross-project retrieval with domain boost)
- **Files**: `minime/config.py` (extend), `minime/core/identity.py` (extend)

**Upgrade #5: Structured Output Strictness**
- Validate JSON against Pydantic schema pre-execution
- If invalid: auto-repair (ask LLM to fix)
- Fallback templates (if repair fails)
- Metrics: % valid, repairs needed
- **Files**: `minime/core/output_validator.py`

**Upgrade #6: Evaluation Harness**
- Benchmark suite: 50 tasks with known good outputs
- Score quality: string sim, structure validity, execution success
- Track: quality over time, cost trend
- Dashboard: `minime eval --latest-10-runs`
- **Files**: `scripts/eval.py`, `minime/eval/harness.py`

**Upgrade #7: Streaming LLM Responses**
- Use OpenAI streaming API (`stream=True`)
- Stream tokens to CLI in real-time
- Collect full response for parsing + tracing
- **Files**: `minime/providers/openai_provider.py` (extend)

**Upgrade #8: PostgreSQL Migration**
- `minime/storage/pgvector_store.py`
- Migration script: `sqlite → postgres`
- Connection pooling (asyncpg)
- Backup strategy
- **Files**: `minime/storage/pgvector_store.py`, `scripts/migrate_pg.py`

**Upgrade #9: Sandboxed Execution**
- Docker container per execution
- `minime/execution/sandbox.py` (docker-py)
- Timeout + resource limits (CPU, RAM, disk)
- Capture output + stderr
- **Files**: `minime/execution/sandbox.py`, `docker/Dockerfile.sandbox`

**Upgrade #10: LoRA Adapters**
- Optional: low-rank adapters on frozen LLM
- Train adapters on user's domain (protein, robotics, etc.)
- Merge adapters at inference time
- Complex; recommend Phase 3
- **Files**: `minime/adapters/lora.py`, `scripts/train_lora.py`

---

## LOCAL DEV SETUP (DAY 1)

### Prerequisites
- Python 3.11+
- pip / poetry
- OpenAI API key (set `OPENAI_API_KEY` env var)
- (Optional) Docker for pgvector

### Quick Start (15 minutes)

```bash
# Clone repo (or init new)
git clone https://github.com/yourusername/minime.git
cd minime

# Create virtual env
python3.11 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install deps
pip install -r requirements.txt

# Initialize MiniMe
python -m minime.main init --name "Your Name"

# Verify setup
python -m minime.main ask "Hello, MiniMe!"

# Inspect trace
ls -la .minime/traces/
cat .minime/traces/*.jsonl | jq .
```

### Repository Structure (After Init)

```
minime/
├── venv/
├── .minime/                (created by `minime init`)
│   ├── config.yaml
│   ├── identity.json
│   ├── allowlist.yaml
│   ├── memory.db
│   └── traces/
├── requirements.txt
├── minime/
│   ├── main.py
│   ├── config.py
│   ├── core/
│   ├── agents/
│   ├── providers/
│   ├── storage/
│   ├── tools/
│   └── models/
└── tests/
```

### Requirements.txt (MVP)

```
# Core
typer==0.9.0
pydantic==2.5.3
python-dotenv==1.0.0

# LLM + Embeddings
openai==1.3.0
asyncio==3.4.3

# Data
numpy==1.24.3
pyyaml==6.0.1

# DB
sqlalchemy==2.0.23

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Dev
black==23.12.0
ruff==0.1.11
mypy==1.7.1
```

### Environment Setup

```bash
# .env (source this in development)
export OPENAI_API_KEY="sk-..."
export MINIME_CONFIG_DIR=".minime"
export MINIME_LOG_LEVEL="debug"

# Load it
set -a; source .env; set +a
```

### First Commands to Try

```bash
# 1. Initialize
minime init --name "Alice" --config-dir .minime

# 2. Import notes (if you have some)
minime import ~/notes/ml --domain ml

# 3. Ask a simple question
minime ask "Plan a Python project structure for a CLI tool" --verbose

# 4. Inspect the trace
minime trace <run-id>  # copy run ID from previous output

# 5. Edit config
minime config edit

# 6. Dry run (no LLM call)
minime ask "Design a database schema" --dry-run
```

### Debugging + Logging

```bash
# Enable debug logging
minime ask "..." --verbose

# Watch traces in real-time
watch -n 1 'ls -lart .minime/traces/ | tail -1'

# Parse trace JSON
cat .minime/traces/{run_id}.jsonl | jq '.data'

# Check token usage
cat .minime/traces/{run_id}.jsonl | jq 'select(.event_type=="llm_call")'
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_context_broker.py -v

# With coverage
pytest --cov=minime

# Watch mode (requires pytest-watch)
ptw
```

### Common Issues + Fixes

| Issue | Fix |
|-------|-----|
| `OPENAI_API_KEY` not found | `export OPENAI_API_KEY="sk-..."` |
| `ModuleNotFoundError: minime` | `pip install -e .` (editable install) |
| SQLite locked | Delete `.minime/memory.db-shm`, `.minime/memory.db-wal` |
| LLM timeout | Increase `--timeout` in config, or use cheaper model |

---

## ASSUMPTIONS (EXPLICIT)

1. **User has OpenAI API key** (or Ollama running locally)
2. **Single-machine development** (day-1 setup; multi-user comes later)
3. **Python 3.11+ available**
4. **SQLite sufficient for MVP** (~100k vectors); pgvector for Phase 2
5. **No fine-tuning of LLMs** (frozen models only; conditioning via mask network)
6. **User approves before any tool execution** (gated automation)
7. **All traces are local** (no cloud sync day-1)
8. **Identity (P_global) is bootstrapped manually** (heuristic mask weights; learned masks Phase 2)
9. **No streaming responses day-1** (full response + trace)
10. **Tool allowlist is maintained manually** (YAML file edit)

---

## PHASE BREAKDOWN (REALISTIC TIMELINE)

| Phase | Weeks | Features | Output |
|-------|-------|----------|--------|
| **Phase 0: Setup** | 0.5 | Repo scaffold, requirements, CI/CD | Buildable codebase |
| **Phase 1: Foundation** | 1-1.5 | CLI, LLM provider, heuristic mask, basic agent, tracing | `minime ask` works end-to-end |
| **Phase 2: Multi-Agent + Memory** | 1.5-2 | Architect/Builder/Critic, memory import, retrieval, approval gate | Full orchestration loop |
| **Phase 3: Polish + Testing** | 0.5-1 | Unit tests, E2E tests, config system, docs | Production-ready MVP |
| **Phase 4: First Upgrade** | 1-2 | Trainable mask MLP, feedback loop | Self-improving personalization |
| **Phases 5-11** | Ongoing | VS Code, reranking, per-project profiles, etc. | Feature-rich system |

**Total MVP: 3-4 weeks with 1 person, 2 weeks with 2 people.**

---

## FILES TO CREATE ON DAY 1

```bash
# Create skeleton (copy-paste ready)
touch minime/__init__.py
touch minime/main.py
touch minime/config.py
touch minime/constants.py

# Core modules
mkdir -p minime/core minime/agents minime/providers minime/storage minime/tools minime/models minime/utils
touch minime/core/{__init__,identity,memory,mask_network,context_broker,orchestrator,tracing}.py
touch minime/agents/{__init__,base,architect,builder,critic,integrator,llm_prompts}.py
touch minime/providers/{__init__,base,openai_provider,ollama_provider,mock_provider}.py
touch minime/storage/{__init__,base,sqlite_store,pgvector_store,migrations}.py
touch minime/tools/{__init__,base,definitions,executor,allowlist}.py
touch minime/models/{__init__,schemas,types}.py
touch minime/utils/{__init__,embeddings,chunking,logger,validators}.py

# Tests
mkdir -p tests
touch tests/{__init__,conftest,test_*.py}

# Scripts
mkdir -p scripts
touch scripts/{init_project.sh,train_mask.py,import_notes.py,inspect_traces.py,benchmark.py}

# Docs
mkdir -p docs
touch docs/{ARCHITECTURE,API,QUICKSTART,DEVELOPMENT,EXAMPLES}.md

# GitHub Actions
mkdir -p .github/workflows
touch .github/workflows/{test,lint}.yaml
```

**Start with `minime/main.py` and `minime/config.py`. Get `minime init` working first.**

---

## FINAL CHECKLIST (MVP READY)

- [ ] Repo scaffold complete
- [ ] All Pydantic models defined (models/schemas.py)
- [ ] CLI entrypoint functional (main.py)
- [ ] Config loading works (config.py + .minime/config.yaml)
- [ ] Identity matrix loadable (identity.py)
- [ ] Heuristic mask generation working (mask_network.py)
- [ ] SQLite store can embed + retrieve (storage/sqlite_store.py)
- [ ] Context broker assembles + budgets (context_broker.py)
- [ ] Orchestrator runs end-to-end (orchestrator.py)
- [ ] At least 1 agent working (architect.py)
- [ ] Tool executor + allowlist functional (tools/executor.py)
- [ ] Traces written + readable (tracing.py)
- [ ] Unit tests pass (tests/)
- [ ] E2E test passes with mocked LLM (tests/test_orchestrator_e2e.py)
- [ ] README + quickstart complete
- [ ] `minime ask` command works without errors

---

**This plan is copy-paste ready. Start with Phase 1, build incrementally, integrate feedback loops. You have a shippable MVP in 3 weeks.**

