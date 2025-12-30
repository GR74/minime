# DBSession - Database Session Manager

## Overview

`session.py` provides the `DBSession` class, which manages database connections and transactions for the MiniMe memory system. It implements proper connection lifecycle management, transaction handling, and WAL (Write-Ahead Logging) mode for better concurrency.

## Purpose

The DBSession class solves several critical database management issues:
- **Connection Reuse**: Prevents creating new connections for every operation
- **Transaction Management**: Provides explicit commit/rollback control
- **Concurrency**: Enables WAL mode for better multi-threaded access
- **Connection Health**: Detects and recovers from stale/closed connections

## Key Components

### DBSession Class

```python
class DBSession:
    def __init__(self, db_path: str)
    async def connect() -> aiosqlite.Connection
    async def execute(query: str, params: tuple = ())
    async def commit()
    async def rollback()
    async def close()
```

## How It Works

### 1. Connection Management

The session maintains a single connection (`_conn`) that is reused across operations:

```python
async def connect(self) -> aiosqlite.Connection:
    # Check if connection exists and is valid
    if self._conn is not None:
        # Validate connection health
        if hasattr(self._conn, '_running') and not self._conn._running:
            self._conn = None  # Reset if stale
    
    # Create new connection if needed
    if self._conn is None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        # Enable WAL mode for better concurrency
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
    
    return self._conn
```

**Key Features:**
- **Lazy Connection**: Connection is created only when needed
- **Health Checking**: Validates connection before reuse
- **WAL Mode**: Enables concurrent reads and writes
- **Row Factory**: Returns rows as dictionary-like objects

### 2. Transaction Control

Unlike auto-commit mode, DBSession requires explicit commits:

```python
session = DBSession(db_path)
conn = await session.connect()
await conn.execute("INSERT INTO ...")
await session.commit()  # Explicit commit required
```

**Why Explicit Commits?**
- **Batch Operations**: Multiple operations can be grouped in a single transaction
- **Atomicity**: All-or-nothing execution for related operations
- **Performance**: Fewer disk writes when batching

### 3. Error Recovery

The session handles connection failures gracefully:

```python
try:
    # Check connection health
    if hasattr(self._conn, '_connection') and self._conn._connection is None:
        self._conn = None  # Reset on failure
except (AttributeError, RuntimeError):
    self._conn = None  # Reset on exception
```

## Usage Examples

### Basic Usage

```python
from minime.memory.session import DBSession

# Create session
session = DBSession("./data/minime.db")

# Connect and execute
conn = await session.connect()
await conn.execute("INSERT INTO vault_nodes VALUES (?, ?)", (node_id, path))
await session.commit()

# Close when done
await session.close()
```

### Batch Operations

```python
session = DBSession(db_path)
conn = await session.connect()

# Multiple operations in one transaction
for node in nodes:
    await conn.execute("INSERT INTO vault_nodes ...", (node.id, node.path))

# Single commit for all operations
await session.commit()
await session.close()
```

### Error Handling

```python
session = DBSession(db_path)
try:
    conn = await session.connect()
    await conn.execute("INSERT ...")
    await session.commit()
except Exception as e:
    await session.rollback()  # Undo changes on error
    raise
finally:
    await session.close()
```

## Integration with AsyncDatabase

The `AsyncDatabase` class uses `DBSession` internally:

```python
class AsyncDatabase:
    def __init__(self, db_path: str):
        self._session = DBSession(db_path)
    
    async def insert_node(self, node: VaultNode):
        # Uses session internally
        await self._session.execute("INSERT ...")
        await self._session.commit()
```

## Technical Details

### WAL Mode (Write-Ahead Logging)

WAL mode improves concurrency by:
- **Concurrent Reads**: Multiple readers can access the database simultaneously
- **Non-Blocking Writes**: Writers don't block readers
- **Better Performance**: Fewer lock contentions

```sql
PRAGMA journal_mode=WAL;      -- Enable WAL
PRAGMA synchronous=NORMAL;    -- Balance safety and speed
```

### Connection Lifecycle

1. **Creation**: Connection created on first `connect()` call
2. **Reuse**: Same connection reused for subsequent operations
3. **Health Check**: Connection validated before each use
4. **Recovery**: Stale connections are automatically reset
5. **Cleanup**: Connection closed explicitly via `close()`

## Best Practices

1. **Always Commit**: Don't forget to call `commit()` after writes
2. **Use Transactions**: Group related operations in a single transaction
3. **Close Explicitly**: Call `close()` when done (or use context manager)
4. **Error Handling**: Use `rollback()` on exceptions
5. **Connection Reuse**: Reuse the same session for multiple operations

## Common Issues and Solutions

### Issue: Changes Not Persisting

**Problem**: Data not saved after `execute()`

**Solution**: Call `commit()` after writes:
```python
await session.execute("INSERT ...")
await session.commit()  # Don't forget this!
```

### Issue: Connection Errors

**Problem**: "Connection is closed" errors

**Solution**: The session automatically detects and recovers from stale connections. If issues persist, explicitly close and recreate:
```python
await session.close()
session = DBSession(db_path)  # Create new session
```

### Issue: Concurrent Access

**Problem**: Database locked errors

**Solution**: WAL mode is enabled automatically, which handles concurrent access better. Ensure you're using WAL mode:
```python
await conn.execute("PRAGMA journal_mode=WAL")
```

## Summary

`DBSession` is a critical component that provides:
- ✅ Reliable connection management
- ✅ Explicit transaction control
- ✅ Automatic error recovery
- ✅ Better concurrency with WAL mode
- ✅ Simple, clean API

It's the foundation for all database operations in the MiniMe memory system, ensuring data integrity and performance.

