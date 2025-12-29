"""Note summarizer for auto-generating Obsidian notes from AI conversations."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.vault import VaultIndexer
from minime.schemas import MiniMeConfig, VaultNode


class NoteSummarizer:
    """Generates Obsidian notes from AI task conversations."""

    def __init__(
        self,
        vault_path: str,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        config: MiniMeConfig,
        provider: Any = None,  # LLMProvider - using Any for now since providers not implemented
    ):
        """
        Initialize NoteSummarizer.

        Args:
            vault_path: Path to Obsidian vault
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance
            config: MiniMeConfig instance
            provider: LLMProvider instance (optional, will use mock if not provided)
        """
        self.vault_path = Path(vault_path).expanduser()
        self.db = db
        self.embedding_model = embedding_model
        self.config = config
        self.provider = provider

    async def generate_note(
        self,
        task_query: str,
        conversation_history: List[Dict[str, Any]],
        final_output: str,
    ) -> Optional[VaultNode]:
        """
        Generate an Obsidian note from a task conversation.

        Args:
            task_query: Original user query
            conversation_history: List of messages (user/assistant exchanges)
            final_output: Final result/output from the task

        Returns:
            VaultNode if successful, None otherwise
        """
        try:
            # Generate note content
            note_content = await self._generate_note_content(conversation_history, final_output)

            # Create frontmatter
            title = self._generate_title(task_query, note_content)
            tags = ["ai-memory", "auto-generated"]
            domain = self._extract_domain(task_query)

            frontmatter = self._create_frontmatter(title, tags, domain)

            # Save note to vault
            filename = self._generate_filename(title)
            note_path = await self._save_note_to_vault(frontmatter, note_content, filename)

            # Index the note
            indexer = VaultIndexer(
                vault_path=str(self.vault_path),
                db=self.db,
                embedding_model=self.embedding_model,
            )

            # Process the newly created file
            node = await indexer._process_file(Path(note_path))

            return node

        except Exception as e:
            print(f"Error generating note: {e}")
            return None

    async def _generate_note_content(
        self, conversation: List[Dict[str, Any]], output: str, max_retries: int = 3
    ) -> str:
        """
        Generate note content using LLM or fallback template.

        Args:
            conversation: Conversation history
            output: Final output
            max_retries: Maximum retry attempts

        Returns:
            Generated note content as markdown
        """
        # Build prompt
        prompt = self._build_prompt(conversation, output)

        # Try to use provider if available
        if self.provider and hasattr(self.provider, "generate"):
            for attempt in range(max_retries):
                try:
                    result = await self.provider.generate(
                        prompt=prompt,
                        system="You are a helpful assistant creating memory notes for an AI system.",
                        model=getattr(self.provider, "model", "gpt-4"),
                        max_tokens=2000,
                        temperature=0.7,
                    )

                    # Extract text from result (handle different result formats)
                    if hasattr(result, "text"):
                        return result.text
                    elif isinstance(result, str):
                        return result
                    elif isinstance(result, dict) and "text" in result:
                        return result["text"]
                    else:
                        # Fallback to template
                        break

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"Warning: Provider failed after {max_retries} attempts: {e}")
                        break

        # Fallback to template-based generation
        return self._generate_fallback_note(conversation, output)

    def _build_prompt(self, conversation: List[Dict[str, Any]], output: str) -> str:
        """
        Build prompt for LLM to generate note.

        Args:
            conversation: Conversation history
            output: Final output

        Returns:
            Prompt string
        """
        # Format conversation
        conv_text = ""
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conv_text += f"{role.capitalize()}: {content}\n\n"

        prompt = f"""You are creating a memory note for an AI assistant system. Based on this conversation and output, create a well-structured Obsidian markdown note.

Conversation:
{conv_text}

Final Output:
{output}

Create a note that includes:
1. Summary of what was discussed/accomplished
2. Key decisions or insights
3. Action items or tasks (if any)
4. Code snippets or solutions (if any)
5. Important context for future reference

Format the note as markdown. Use headings, lists, and code blocks as appropriate. The note should be useful for future reference and help the AI understand context."""
        return prompt

    def _generate_fallback_note(
        self, conversation: List[Dict[str, Any]], output: str
    ) -> str:
        """
        Generate a simple template-based note when LLM is not available.

        Args:
            conversation: Conversation history
            output: Final output

        Returns:
            Template-based note content
        """
        note_lines = ["# Summary\n"]

        # Add conversation summary
        note_lines.append("## Conversation\n")
        for msg in conversation[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]  # Truncate long messages
            note_lines.append(f"- **{role.capitalize()}**: {content}\n")

        # Add output
        note_lines.append("\n## Output\n")
        note_lines.append(f"```\n{output[:500]}\n```\n")  # Truncate if too long

        # Add timestamp
        note_lines.append(f"\n*Generated: {datetime.now().isoformat()}*\n")

        return "".join(note_lines)

    def _generate_title(self, task_query: str, note_content: str) -> str:
        """
        Generate note title from task query or note content.

        Args:
            task_query: Original task query
            note_content: Generated note content

        Returns:
            Title string
        """
        # Try to extract title from note content (first heading)
        if note_content:
            first_line = note_content.split("\n")[0]
            if first_line.startswith("# "):
                return first_line[2:].strip()

        # Fallback: use task query (truncated)
        title = task_query[:50].strip()
        if len(task_query) > 50:
            title += "..."

        return title

    def _extract_domain(self, task_query: str) -> Optional[str]:
        """
        Extract domain from task query (simple heuristic).

        Args:
            task_query: Task query text

        Returns:
            Domain string or None
        """
        query_lower = task_query.lower()

        # Simple keyword matching
        if any(word in query_lower for word in ["python", "code", "programming", "script"]):
            return "coding"
        elif any(word in query_lower for word in ["biotech", "biology", "protein", "dna"]):
            return "biotech"
        elif any(word in query_lower for word in ["design", "architecture", "system"]):
            return "architecture"

        return None

    def _create_frontmatter(
        self, title: str, tags: List[str], domain: Optional[str]
    ) -> Dict[str, Any]:
        """
        Create frontmatter dictionary.

        Args:
            title: Note title
            tags: List of tags
            domain: Domain tag (optional)

        Returns:
            Frontmatter dictionary
        """
        frontmatter = {
            "title": title,
            "tags": tags,
            "scope": "global",
            "created_at": datetime.now().isoformat(),
            "source": "ai-task",
        }

        if domain:
            frontmatter["domain"] = domain

        return frontmatter

    def _generate_filename(self, title: str) -> str:
        """
        Generate filename from title.

        Args:
            title: Note title

        Returns:
            Filename string
        """
        # Slugify title
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[-\s]+", "-", slug)
        slug = slug[:50]  # Limit length

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{timestamp}-{slug}.md"

        return filename

    async def _save_note_to_vault(
        self, frontmatter: Dict[str, Any], content: str, filename: str
    ) -> str:
        """
        Save note to vault.

        Args:
            frontmatter: Frontmatter dictionary
            content: Note content
            filename: Filename

        Returns:
            Full path to saved file
        """
        # Create ai-memory directory
        ai_memory_dir = self.vault_path / self.config.ai_memory_path
        ai_memory_dir.mkdir(parents=True, exist_ok=True)

        # Write file with frontmatter
        file_path = ai_memory_dir / filename

        # Format frontmatter as YAML
        import yaml

        frontmatter_yaml = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        note_text = f"---\n{frontmatter_yaml}---\n\n{content}"

        file_path.write_text(note_text, encoding="utf-8")

        return str(file_path)

