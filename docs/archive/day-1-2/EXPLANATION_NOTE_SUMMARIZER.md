# Explanation: `minime/memory/summarizer.py`

This file provides the `NoteSummarizer` class that automatically generates Obsidian notes from AI task conversations, similar to ChatGPT's context memory feature.

## Overview

The `NoteSummarizer` creates persistent memory notes after each AI task completion:
1. **Generates note content** using LLM or template
2. **Creates frontmatter** with metadata
3. **Saves to vault** in `ai-memory/` folder
4. **Auto-indexes** the note for future retrieval

Think of it as "taking notes" for the AI so it remembers past conversations.

---

## File Structure

```python
class NoteSummarizer:
    # Main method
    - generate_note(task_query, conversation_history, final_output)  # Generate note
    
    # Content generation
    - _generate_note_content(conversation, output)  # LLM or template
    - _build_prompt(conversation, output)           # LLM prompt
    - _generate_fallback_note(conversation, output) # Template-based
    
    # Metadata
    - _generate_title(task_query, note_content)     # Extract title
    - _extract_domain(task_query)                   # Detect domain
    - _create_frontmatter(title, tags, domain)      # Build frontmatter
    
    # File operations
    - _generate_filename(title)                     # Create filename
    - _save_note_to_vault(frontmatter, content, filename)  # Save file
```

---

## Class: `NoteSummarizer`

### Constructor: `__init__(vault_path, db, embedding_model, config, provider=None)`

**Parameters:**
- `vault_path`: Path to Obsidian vault
- `db`: `AsyncDatabase` instance
- `embedding_model`: `EmbeddingModel` instance
- `config`: `MiniMeConfig` instance
- `provider`: LLMProvider instance (optional, uses template if None)

**Example:**
```python
summarizer = NoteSummarizer(
    vault_path="~/my-vault",
    db=db,
    embedding_model=embedding_model,
    config=config,
    provider=llm_provider  # Optional
)
```

---

## Method: `async generate_note(...) -> Optional[VaultNode]`

**Purpose**: Generate an Obsidian note from a task conversation.

**Parameters:**
- `task_query`: Original user query (e.g., "Create a Python script")
- `conversation_history`: List of messages (user/assistant exchanges)
- `final_output`: Final result/output from the task

**Returns**: `VaultNode` if successful, `None` on error

**Workflow:**
1. Generate note content (LLM or template)
2. Create frontmatter (title, tags, domain)
3. Save note to vault (`ai-memory/` folder)
4. Index the note (so it's searchable)
5. Return `VaultNode`

**Example:**
```python
conversation = [
    {"role": "user", "content": "Create a Python script to process CSV files"},
    {"role": "assistant", "content": "I'll create a script that reads CSV files..."}
]

note = await summarizer.generate_note(
    task_query="Create a Python script to process CSV files",
    conversation_history=conversation,
    final_output="Script created successfully at ./process_csv.py"
)

print(f"Note created: {note.path}")
```

---

## Method: `async _generate_note_content(...) -> str`

**Purpose**: Generate note content using LLM or fallback template.

**What it does:**
1. Builds prompt from conversation and output
2. Tries to use LLM provider (if available)
3. Falls back to template if LLM fails or unavailable

**LLM Usage:**
```python
if self.provider and hasattr(self.provider, "generate"):
    result = await self.provider.generate(
        prompt=prompt,
        system="You are a helpful assistant creating memory notes...",
        model="gpt-4",
        max_tokens=2000,
        temperature=0.7
    )
    return result.text  # Extract text from result
```

**Retry Logic:**
- Attempts up to 3 times
- Exponential backoff between retries
- Falls back to template on failure

**Template Fallback:**
```python
# If LLM unavailable or fails
return self._generate_fallback_note(conversation, output)
```

---

## Method: `_build_prompt(conversation, output) -> str`

**Purpose**: Build prompt for LLM to generate structured note.

**Prompt Template:**
```
You are creating a memory note for an AI assistant system. Based on this conversation and output, create a well-structured Obsidian markdown note.

Conversation:
{formatted_conversation}

Final Output:
{output}

Create a note that includes:
1. Summary of what was discussed/accomplished
2. Key decisions or insights
3. Action items or tasks (if any)
4. Code snippets or solutions (if any)
5. Important context for future reference

Format the note as markdown. Use headings, lists, and code blocks as appropriate.
```

**Conversation Formatting:**
```python
conv_text = ""
for msg in conversation:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    conv_text += f"{role.capitalize()}: {content}\n\n"
```

---

## Method: `_generate_fallback_note(...) -> str`

**Purpose**: Generate simple template-based note when LLM unavailable.

**Template Structure:**
```markdown
# Summary

## Conversation

- **User**: [last 5 messages]
- **Assistant**: [last 5 messages]

## Output

```
{output (truncated to 500 chars)}
```

*Generated: {timestamp}*
```

**Limitations:**
- Simple format (not as structured as LLM-generated)
- Truncates long content
- Basic formatting

**Use case**: Testing, offline mode, or when LLM unavailable.

---

## Method: `_generate_title(task_query, note_content) -> str`

**Purpose**: Generate note title from task query or note content.

**Strategy:**
1. Try to extract title from note content (first heading `# Title`)
2. Fallback: Use task query (truncated to 50 chars)

**Example:**
```python
# If note content has heading
note_content = "# Python CSV Processing Script\n\nThis note..."
title = _generate_title("Create script", note_content)
# Returns: "Python CSV Processing Script"

# If no heading
title = _generate_title("Create a Python script to process CSV files", "")
# Returns: "Create a Python script to process CSV files" (truncated)
```

---

## Method: `_extract_domain(task_query) -> Optional[str]`

**Purpose**: Extract domain from task query using simple keyword matching.

**Keywords:**
- **"coding"**: python, code, programming, script
- **"biotech"**: biotech, biology, protein, dna
- **"architecture"**: design, architecture, system

**Example:**
```python
domain = _extract_domain("Create a Python script")
# Returns: "coding"

domain = _extract_domain("Design a system architecture")
# Returns: "architecture"
```

**Future improvement**: Could use LLM or more sophisticated NLP to detect domain.

---

## Method: `_create_frontmatter(title, tags, domain) -> Dict`

**Purpose**: Create frontmatter dictionary for note.

**Default Fields:**
```yaml
title: {title}
tags: ["ai-memory", "auto-generated"]
scope: "global"
created_at: {iso_timestamp}
source: "ai-task"
domain: {domain}  # If detected
```

**Example:**
```python
frontmatter = {
    "title": "Python CSV Processing Script",
    "tags": ["ai-memory", "auto-generated"],
    "scope": "global",
    "created_at": "2024-01-15T10:30:00",
    "source": "ai-task",
    "domain": "coding"
}
```

---

## Method: `_generate_filename(title) -> str`

**Purpose**: Generate filename from title.

**Format**: `{timestamp}-{slugified-title}.md`

**Steps:**
1. Slugify title (remove special chars, lowercase, replace spaces with hyphens)
2. Truncate to 50 characters
3. Add timestamp prefix
4. Add `.md` extension

**Example:**
```python
title = "Python CSV Processing Script"
filename = _generate_filename(title)
# Returns: "2024-01-15-10-30-00-python-csv-processing-script.md"
```

---

## Method: `async _save_note_to_vault(...) -> str`

**Purpose**: Save note to vault with frontmatter.

**Location**: `{vault_path}/{ai_memory_path}/{filename}`

**Default**: `vault/ai-memory/YYYY-MM-DD-HH-MM-SS-title.md`

**Format:**
```markdown
---
title: Python CSV Processing Script
tags: [ai-memory, auto-generated]
scope: global
created_at: 2024-01-15T10:30:00
source: ai-task
domain: coding
---

# Note Content

This is the generated note content...
```

**What it does:**
1. Creates `ai-memory/` directory if it doesn't exist
2. Formats frontmatter as YAML
3. Writes file with frontmatter + content
4. Returns full file path

---

## Integration with Indexer

After saving, the note is automatically indexed:

```python
# In generate_note()
note_path = await self._save_note_to_vault(...)

# Index the note
indexer = VaultIndexer(
    vault_path=str(self.vault_path),
    db=self.db,
    embedding_model=self.embedding_model
)

node = await indexer._process_file(Path(note_path))
# Note is now searchable!
```

---

## Key Concepts

### 1. Auto-Generated Notes

Notes are created automatically after tasks:
- No user intervention needed
- Builds memory over time
- Similar to ChatGPT's context memory

### 2. LLM vs Template

**LLM-generated** (when provider available):
- Structured and coherent
- Better summaries
- Context-aware

**Template-generated** (fallback):
- Simple format
- Works offline
- Always available

### 3. Auto-Indexing

Generated notes are immediately indexed:
- Available for retrieval in next task
- Computes embeddings
- Creates similarity proposals

### 4. Domain Detection

Simple keyword matching detects domain:
- Tags notes appropriately
- Enables domain filtering
- Could be improved with NLP

### 5. File Organization

Notes saved to `ai-memory/` folder:
- Keeps auto-generated notes separate
- Easy to identify
- Can be organized further

---

## Configuration

Configurable via `MiniMeConfig`:

```python
class MiniMeConfig:
    ai_memory_path: str = "ai-memory"           # Subfolder for notes
    auto_generate_notes: bool = True            # Enable/disable
    note_generation_provider: str = "mock"      # Provider name
    use_local_llm_for_notes: bool = False       # Use local LLM
    local_llm_model: str = "llama3"            # Local model name
```

**Modes:**
1. **Fully Local**: `use_local_llm_for_notes=True`, `provider="ollama"`
2. **API**: `use_local_llm_for_notes=False`, `provider="openai"`
3. **Template**: No provider (always uses template)

---

## Example Workflow

```python
# 1. User completes task
task_query = "Create a Python script"
conversation = [...]  # User/assistant exchanges
final_output = "Script created at ./script.py"

# 2. Generate note
summarizer = NoteSummarizer(...)
note = await summarizer.generate_note(
    task_query=task_query,
    conversation_history=conversation,
    final_output=final_output
)

# 3. Note is saved and indexed
# Location: vault/ai-memory/2024-01-15-10-30-00-create-python-script.md
# Now searchable in next task!
```

---

## Future Enhancements

### 1. Better Domain Detection

Use LLM or NLP to detect domain:
```python
domain = await llm_classify_domain(task_query)
```

### 2. Note Templates

Customizable templates per domain:
```yaml
templates:
  coding:
    sections: [summary, code, decisions, next-steps]
  research:
    sections: [findings, sources, questions, conclusions]
```

### 3. Note Quality Scoring

Rate generated notes and improve prompts:
```python
quality_score = await rate_note_quality(note_content)
if quality_score < 0.7:
    # Improve prompt or regenerate
```

### 4. Note Linking

Automatically link related notes:
```python
# Find similar notes and create wikilinks
similar_notes = await find_similar_notes(note)
note_content += "\n\n## Related Notes\n" + format_links(similar_notes)
```

---

## Summary

The `NoteSummarizer` is the **memory creation layer** for MiniMe:

1. **Generates notes** from AI conversations
2. **Creates structured content** (LLM or template)
3. **Saves to vault** in organized folder
4. **Auto-indexes** for immediate retrieval
5. **Builds persistent memory** over time

Without the summarizer, MiniMe would have no way to remember past conversations. Each task completion creates a memory note, building a knowledge base that makes the AI smarter over time.

