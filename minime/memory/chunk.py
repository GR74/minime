"""Note chunking strategy for splitting long notes into overlapping chunks."""


def chunk_note(note_text: str, max_tokens: int = 512, overlap: int = 128) -> list[str]:
    """
    Split note body into overlapping chunks.

    Args:
        note_text: Full note text (body only, no frontmatter)
        max_tokens: Maximum tokens per chunk (default: 512)
        overlap: Number of tokens to overlap between chunks (default: 128)

    Returns:
        List of chunk strings
    """
    if not note_text or not note_text.strip():
        return []

    # Simple tokenization: split on whitespace
    # For MVP, we use word-based tokenization
    # In production, could use proper tokenizer (tiktoken, etc.)
    tokens = note_text.split()

    # If note is shorter than max_tokens, return as single chunk
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]

    chunks = []
    step_size = max_tokens - overlap

    # Create overlapping chunks
    for i in range(0, len(tokens), step_size):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = " ".join(chunk_tokens)

        # Try to preserve sentence boundaries
        # If we're not at the start, try to start at a sentence boundary
        if i > 0 and chunk_text:
            # Look for sentence endings (., !, ?) near the start
            # This is a simple heuristic - could be improved
            first_period = chunk_text.find(". ", 0, min(100, len(chunk_text)))
            if first_period > 0:
                # Start from after the period
                chunk_text = chunk_text[first_period + 2 :]

        if chunk_text.strip():
            chunks.append(chunk_text.strip())

    # Ensure we don't lose the end
    if len(tokens) > 0:
        last_chunk_start = max(0, len(tokens) - max_tokens)
        last_chunk = " ".join(tokens[last_chunk_start:])
        if last_chunk.strip() and (not chunks or chunks[-1] != last_chunk.strip()):
            chunks.append(last_chunk.strip())

    return chunks if chunks else [note_text]

