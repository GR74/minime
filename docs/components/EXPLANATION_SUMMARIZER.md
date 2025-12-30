# NoteSummarizer - Auto-Generated Note Creation

## Overview

`summarizer.py` provides the `NoteSummarizer` class, which automatically generates Obsidian notes from AI task conversations. This enables the system to learn from interactions and build a knowledge base over time.

## Purpose

The NoteSummarizer addresses the need to:
- **Capture Knowledge**: Save important information from AI conversations
- **Build Memory**: Create searchable notes for future reference
- **Learn Patterns**: Track decisions and solutions over time
- **Auto-Organization**: Automatically structure and tag generated notes

## Key Components

### NoteSummarizer Class

```python
class NoteSummarizer:
    def __init__(
        self,
        vault_path: str,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        config: MiniMeConfig,
        provider: Any = None,
    )
    
    async def generate_note(
        task_query: str,
        conversation_history: List[Dict[str, Any]],
        final_output: str,
    ) -> Optional[VaultNode]
```

## How It Works

### 1. Note Generation Process

```python
async def generate_note(
    self,
    task_query: str,
    conversation_history: List[Dict[str, Any]],
    final_output: str,
) -> Optional[VaultNode]:
    # Step 1: Generate note content using LLM or template
    note_content = await self._generate_note_content(conversation_history, final_output)
    
    # Step 2: Create frontmatter
    title = self._generate_title(task_query, note_content)
    tags = ["ai-memory", "auto-generated"]
    domain = self._extract_domain(task_query)
    frontmatter = self._create_frontmatter(title, tags, domain)
    
    # Step 3: Save note to vault
    filename = self._generate_filename(title)
    note_path = await self._save_note_to_vault(frontmatter, note_content, filename)
    
    # Step 4: Index the note
    indexer = VaultIndexer(vault_path, db, embedding_model)
    node = await indexer._process_file(Path(note_path))
    
    return node
```

### 2. Content Generation

The summarizer uses an LLM (if available) or falls back to a template:

```python
async def _generate_note_content(
    self, conversation: List[Dict[str, Any]], output: str, max_retries: int = 3
) -> str:
    # Build prompt
    prompt = self._build_prompt(conversation, output)
    
    # Try to use provider if available
    if self.provider and hasattr(self.provider, "generate"):
        for attempt in range(max_retries):
            try:
                result = await self.provider.generate(
                    prompt=prompt,
                    system="You are a helpful assistant creating memory notes...",
                    model=getattr(self.provider, "model", "gpt-4"),
                    max_tokens=2000,
                    temperature=0.7,
                )
                return result.text
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    break
    
    # Fallback to template-based generation
    return self._generate_fallback_note(conversation, output)
```

**LLM Prompt Structure**:
```
You are creating a memory note for an AI assistant system. Based on this conversation and output, create a well-structured Obsidian markdown note.

Conversation:
[formatted conversation history]

Final Output:
[final result]

Create a note that includes:
1. Summary of what was discussed/accomplished
2. Key decisions or insights
3. Action items or tasks (if any)
4. Code snippets or solutions (if any)
5. Important context for future reference
```

### 3. Template Fallback

When LLM is unavailable, uses a simple template:

```python
def _generate_fallback_note(
    self, conversation: List[Dict[str, Any]], output: str
) -> str:
    note_lines = ["# Summary\n"]
    
    # Add conversation summary
    note_lines.append("## Conversation\n")
    for msg in conversation[-5:]:  # Last 5 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]  # Truncate
        note_lines.append(f"- **{role.capitalize()}**: {content}\n")
    
    # Add output
    note_lines.append("\n## Output\n")
    note_lines.append(f"```\n{output[:500]}\n```\n")
    
    # Add timestamp
    note_lines.append(f"\n*Generated: {datetime.now().isoformat()}*\n")
    
    return "".join(note_lines)
```

### 4. Metadata Extraction

Automatically extracts metadata from the task:

```python
def _extract_domain(self, task_query: str) -> Optional[str]:
    query_lower = task_query.lower()
    
    # Simple keyword matching
    if any(word in query_lower for word in ["python", "code", "programming", "script"]):
        return "coding"
    elif any(word in query_lower for word in ["biotech", "biology", "protein", "dna"]):
        return "biotech"
    elif any(word in query_lower for word in ["design", "architecture", "system"]):
        return "architecture"
    
    return None
```

### 5. File Naming

Generates filenames from titles:

```python
def _generate_filename(self, title: str) -> str:
    # Slugify title
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[-\s]+", "-", slug)
    slug = slug[:50]  # Limit length
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{timestamp}-{slug}.md"
    
    return filename
```

**Example**:
- Title: "How to Structure Python Code"
- Filename: `2025-12-29-19-30-45-how-to-structure-python-code.md`

## Usage Examples

### Basic Usage

```python
from minime.memory.summarizer import NoteSummarizer
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.config import load_config

# Initialize
db = AsyncDatabase("./data/minime.db")
model = EmbeddingModel()
config = load_config()

summarizer = NoteSummarizer(
    vault_path="~/obsidian-vault",
    db=db,
    embedding_model=model,
    config=config,
    provider=None  # Will use template fallback
)

# Generate note from conversation
conversation = [
    {"role": "user", "content": "How should I structure my Python project?"},
    {"role": "assistant", "content": "Use a modular structure with separate packages..."},
]
final_output = "Created project structure with src/, tests/, docs/"

node = await summarizer.generate_note(
    task_query="How should I structure my Python project?",
    conversation_history=conversation,
    final_output=final_output
)
```

### With LLM Provider

```python
from minime.providers.mock import MockProvider

# Use LLM provider for better note generation
provider = MockProvider()  # Or real provider

summarizer = NoteSummarizer(
    vault_path="~/obsidian-vault",
    db=db,
    embedding_model=model,
    config=config,
    provider=provider
)

# LLM will generate structured note
node = await summarizer.generate_note(
    task_query="...",
    conversation_history=conversation,
    final_output=output
)
```

## Integration with VaultIndexer

After generating a note, it's automatically indexed:

```python
# In generate_note()
note_path = await self._save_note_to_vault(frontmatter, note_content, filename)

# Index the note
indexer = VaultIndexer(
    vault_path=str(self.vault_path),
    db=self.db,
    embedding_model=self.embedding_model,
)

# Process the newly created file
node = await indexer._process_file(Path(note_path))
```

**Benefits**:
- ✅ Note is immediately searchable
- ✅ Embeddings computed automatically
- ✅ Graph connections created
- ✅ Fully integrated with memory system

## Configuration

The summarizer uses config settings:

```python
# From MiniMeConfig
ai_memory_path: str = "ai-memory"  # Subfolder for auto-generated notes
auto_generate_notes: bool = True   # Whether to auto-generate notes
note_generation_provider: str = "mock"  # Provider for note generation
```

**Note Storage**:
- Notes saved to: `vault_path/ai-memory/`
- Organized by timestamp and title
- Automatically tagged with `["ai-memory", "auto-generated"]`

## Design Decisions

### 1. LLM with Fallback

**Decision**: Use LLM if available, template otherwise

**Rationale**:
- **Quality**: LLM generates better structured notes
- **Reliability**: Template ensures notes are always created
- **Flexibility**: Works with or without LLM provider

### 2. Automatic Indexing

**Decision**: Index notes immediately after creation

**Rationale**:
- **Immediate Searchability**: Notes available right away
- **Consistency**: Same indexing process as manual notes
- **Integration**: Fully integrated with memory system

### 3. Domain Extraction

**Decision**: Simple keyword matching for domain

**Rationale**:
- **Simple**: No complex NLP needed
- **Fast**: Quick domain classification
- **Extensible**: Can be improved with better classifiers

### 4. Timestamp in Filename

**Decision**: Include timestamp in filename

**Rationale**:
- **Uniqueness**: Prevents filename conflicts
- **Chronological**: Easy to sort by creation time
- **Traceability**: Know when note was created

## Best Practices

1. **Use LLM When Available**: Better note quality
2. **Review Generated Notes**: Ensure accuracy
3. **Customize Prompts**: Adjust for your use case
4. **Organize by Domain**: Use domain tags for filtering
5. **Regular Cleanup**: Remove outdated notes

## Common Issues

### Issue: Notes Not Generated

**Problem**: `generate_note()` returns None

**Solutions**:
- Check vault path exists
- Verify write permissions
- Check error logs
- Ensure provider is working (if using LLM)

### Issue: Poor Note Quality

**Problem**: Generated notes are not useful

**Solutions**:
- Use LLM provider instead of template
- Improve prompt structure
- Provide better conversation history
- Add more context to final_output

### Issue: Duplicate Notes

**Problem**: Same note generated multiple times

**Solutions**:
- Check for existing notes before generating
- Use content hashing for deduplication
- Improve task query uniqueness

## Future Enhancements

1. **Better Domain Classification**: Use embeddings or NLP
2. **Note Merging**: Merge related notes
3. **Template Customization**: User-defined templates
4. **Quality Scoring**: Rate note quality
5. **Selective Generation**: Only generate for important conversations

## Summary

The `NoteSummarizer` provides:

- ✅ Automatic note generation from conversations
- ✅ LLM-powered or template-based content
- ✅ Automatic indexing and integration
- ✅ Domain classification and tagging
- ✅ Organized file structure

It's the component that enables MiniMe to learn from interactions and build a searchable knowledge base over time.

