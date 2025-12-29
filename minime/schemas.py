"""Core data schemas for MiniMe system."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# ============================================================================
# Identity Layer
# ============================================================================


class IdentityPrinciple(BaseModel):
    """A single principle in the identity matrix P_global."""

    id: str
    name: str
    description: str
    vector: List[float] = Field(default_factory=list)  # embedding
    magnitude: float = 1.0  # importance weight
    decay_rate: float = 0.05  # how fast it adapts (0.0 = frozen, 1.0 = plastic)
    scope: str = "global"  # "global" | "domain:biotech" | "task:protein_design"
    tags: List[str] = Field(default_factory=list)  # ["architecture", "rigor", "clarity"]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class GlobalIdentityMatrix(BaseModel):
    """P_global: all identity principles."""

    principles: List[IdentityPrinciple] = Field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary mapping principle IDs to their embedding vectors."""
        return {p.id: p.vector for p in self.principles}

    def get_principle(self, principle_id: str) -> Optional[IdentityPrinciple]:
        """Get a principle by ID."""
        for p in self.principles:
            if p.id == principle_id:
                return p
        return None

    def add_principle(self, principle: IdentityPrinciple) -> None:
        """Add a new principle to the matrix."""
        self.principles.append(principle)

    def update_principle(self, principle_id: str, updates: Dict[str, Any]) -> bool:
        """Update a principle with new values."""
        principle = self.get_principle(principle_id)
        if principle is None:
            return False
        for key, value in updates.items():
            if hasattr(principle, key):
                setattr(principle, key, value)
        principle.updated_at = datetime.now()
        return True


# ============================================================================
# Memory Layer (Obsidian Graph)
# ============================================================================


class VaultNode(BaseModel):
    """One note in Obsidian vault."""

    node_id: str  # unique ID (hash of path)
    path: str  # relative path in vault
    title: str
    frontmatter: Dict[str, Any] = Field(default_factory=dict)  # parsed YAML front matter
    tags: List[str] = Field(default_factory=list)  # from frontmatter + inline tags
    domain: Optional[str] = None  # extracted or set by user
    scope: str = "global"  # "global" | "project:myproj"
    links: List[str] = Field(default_factory=list)  # wikilinks [[...]]
    backlinks: List[str] = Field(default_factory=list)  # incoming links
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    embedding_ref: Optional[str] = None  # pointer to embeddings table


class GraphEdge(BaseModel):
    """Explicit or proposed edge in the graph."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # "wikilink" | "backlink" | "similar" | "related"
    weight: float = 1.0  # 0.0 to 1.0
    rationale: str = ""  # why this edge exists
    confidence: float = 1.0  # 0.0 to 1.0 (for proposed edges)
    is_approved: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None


class GraphUpdateProposal(BaseModel):
    """Proposal to add/remove edges."""

    proposal_id: str
    edges_to_add: List[GraphEdge] = Field(default_factory=list)
    edges_to_remove: List[str] = Field(default_factory=list)  # edge IDs
    confidence: float = 0.5
    requires_user_approval: bool = True
    rationale: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryChunk(BaseModel):
    """One chunk of a note (for retrieval)."""

    chunk_id: str
    node_id: str
    content: str
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # domain, tags, scope, etc.
    position: int = 0  # byte offset in note


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

    chunks: List[MemoryChunk] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)  # similarity + graph + recency weighting
    graph_traversal_depth: int = 0
    blocked_count: int = 0  # chunks blocked by scope/security
    context_tokens_used: int = 0
    is_cold_start: bool = False  # True if vault is empty, no memory available


# ============================================================================
# Masking System
# ============================================================================


class MaskWeights(BaseModel):
    """Output of mask network: controls retrieval, generation, routing."""

    # Retrieval
    retrieval_k: int = 5  # how many results to fetch
    retrieval_min_similarity: float = 0.5  # threshold
    block_weights: Dict[str, float] = Field(default_factory=dict)  # per-block weights in context

    # Generation
    temperature: float = 0.7
    verbosity: int = 3  # 1=terse, 5=verbose
    rigor: int = 3  # 1=loose, 5=strict

    # Agent routing
    agent_routing_bias: Dict[str, float] = Field(default_factory=dict)  # {"architect": 0.8, "builder": 0.2, ...}

    # Graph
    graph_proximity_weight: float = 0.2
    graph_max_hops: int = 2

    # Applied masks
    global_mask_strength: float = 1.0
    domain_mask_strength: float = 0.0
    task_mask_strength: float = 0.0
    agent_mask_strength: float = 0.0


class MaskNetworkInput(BaseModel):
    """Input to mask network MLP."""

    z_identity: List[float]  # P_global embedding
    z_task: List[float]  # current task embedding
    z_domain: List[float]  # domain embedding
    agent_type: str


class MaskNetworkOutput(BaseModel):
    """Output of mask network."""

    weights: MaskWeights
    debug_info: Dict[str, Any] = Field(default_factory=dict)


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
    io_schema: Dict[str, Any] = Field(default_factory=dict)  # input schema + output schema
    system_prompt: str
    model: str = "gpt-4"  # "gpt-4" | "claude-3-sonnet" | etc.
    tools_allowed: List[str] = Field(default_factory=list)  # tool names
    routing_hints: Optional[Dict[str, float]] = None  # when to prefer this agent
    constraints: List[str] = Field(default_factory=list)  # what this agent must NOT do
    temperature: float = 0.7
    max_tokens: int = 2000
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None


class ToolDefinition(BaseModel):
    """Definition of a tool agents can call."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)  # JSON schema
    required: List[str] = Field(default_factory=list)
    allowlist_tags: List[str] = Field(default_factory=list)  # ["read", "write", "execute"]
    requires_approval: bool = True


class ToolCall(BaseModel):
    """One tool call from an agent."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of tool execution."""

    success: bool
    output: str = ""
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
    justification: str = ""
    risk_level: str = "medium_risk"  # "safe" | "low_risk" | "medium_risk" | "high_risk"
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None


# ============================================================================
# Tracing & Feedback
# ============================================================================


class TraceEvent(BaseModel):
    """One event in a trace."""

    event_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str  # "retrieve" | "mask_apply" | "agent_output" | "action_propose" | "tool_execute"
    agent_name: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


class RunTrace(BaseModel):
    """Complete trace of one run."""

    run_id: str
    task_query: str
    domain: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    events: List[TraceEvent] = Field(default_factory=list)
    token_usage: Dict[str, int] = Field(default_factory=dict)  # {"prompt": 100, "completion": 50}
    final_output: Optional[str] = None
    success: bool = False


class FeedbackCorrectionPair(BaseModel):
    """User feedback for learning."""

    feedback_id: str
    run_id: str
    original_output: str
    corrected_output: Optional[str] = None
    delta: Optional[str] = None  # diff or edit
    feedback_type: str  # "accept" | "reject" | "edit"
    scope: str = "global"  # "global" | "domain" | "task"
    rationale: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Configuration
# ============================================================================


class MiniMeConfig(BaseModel):
    """Top-level config."""

    vault_path: str  # path to Obsidian vault
    db_path: str  # path to SQLite
    embedding_model: str = "all-MiniLM-L6-v2"  # "all-MiniLM-L6-v2" | "openai"
    embedding_cache_size: int = 100  # MB
    default_provider: str = "mock"  # "openai" | "mock" | "anthropic"
    trace_dir: str = "./logs/"  # ./logs/
    config_dir: str = "./config/"  # ./config/
    max_context_tokens: int = 4000  # per-generation limit
    enable_offline_training: bool = False
    offline_training_interval_hours: int = 24
    # Risk-based execution settings
    safe_paths: List[str] = Field(default_factory=lambda: ["./outputs/", "./docs/", "./tmp/"])
    system_paths: List[str] = Field(default_factory=lambda: ["/usr/", "/etc/", "/bin/"])
    allowlisted_commands: List[str] = Field(default_factory=lambda: ["git status", "git log", "ls", "pwd"])
    auto_approve_safe: bool = True  # auto-execute safe actions
    low_risk_auto_delay_sec: float = 2.0  # delay before auto-executing low-risk actions
    # AI Memory (Auto-Generated Notes)
    ai_memory_path: str = "ai-memory"  # Subfolder for auto-generated notes
    auto_generate_notes: bool = True  # Whether to auto-generate notes after tasks
    note_generation_provider: str = "mock"  # "mock" | "ollama" | "openai" | "anthropic"
    # Local vs API Configuration
    use_local_embeddings: bool = True  # Always use local (sentence-transformers)
    use_local_llm_for_notes: bool = False  # Use local LLM (Ollama) for note generation
    local_llm_model: str = "llama3"  # Model name for Ollama
    local_llm_url: str = "http://localhost:11434"  # Ollama API URL
    # Similarity Search Configuration
    similarity_threshold: float = 0.7  # Threshold for similarity proposals
    auto_approve_confidence: float = 0.9  # Auto-approve proposals above this confidence
    max_similarity_proposals: int = 10  # Max proposals per note
    # Chunking Configuration
    chunk_max_tokens: int = 512  # Max tokens per chunk
    chunk_overlap: int = 128  # Overlap between chunks

    @classmethod
    def load_from_file(cls, path: str) -> "MiniMeConfig":
        """Load configuration from YAML file."""
        import yaml
        from pathlib import Path

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Merge with defaults
        defaults = cls.model_validate({})
        if data:
            # Update defaults with loaded data
            for key, value in data.items():
                if hasattr(defaults, key):
                    setattr(defaults, key, value)

        return defaults

    def save_to_file(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from pathlib import Path

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, excluding None values
        data = self.model_dump(exclude_none=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

