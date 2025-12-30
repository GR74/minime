"""Database session manager for connection pooling and transaction management."""

from typing import Optional
from datetime import datetime
import aiosqlite
import json


class DBSession:
    """
    Database session manager with proper connection lifecycle.
    
    Fixes:
    - No auto-commit (explicit commits required)
    - Proper connection reuse
    - WAL mode for better concurrency
    
    Usage:
        session = DBSession(db_path)
        conn = await session.connect()
        await conn.execute("INSERT ...")
        await session.commit()
        await session.close()
    """

    def __init__(self, db_path: str):
        """
        Initialize DBSession.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> aiosqlite.Connection:
        """
        Get or create connection.
        
        Returns:
            aiosqlite.Connection
        """
        # #region agent log
        try:
            with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "connect() called", "data": {"has_conn": self._conn is not None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
        except: pass
        # #endregion
        
        # Check if connection is stale/closed and reset if needed
        if self._conn is not None:
            try:
                # #region agent log
                try:
                    with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        has_running = hasattr(self._conn, '_running')
                        has_connection = hasattr(self._conn, '_connection')
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "Checking connection health", "data": {"has_running": has_running, "has_connection": has_connection, "running_value": getattr(self._conn, '_running', None) if has_running else None, "connection_value": str(getattr(self._conn, '_connection', None)) if has_connection else None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                except: pass
                # #endregion
                
                # Check if connection is still running/valid
                # aiosqlite connections have _running attribute
                if hasattr(self._conn, '_running') and not self._conn._running:
                    # #region agent log
                    try:
                        with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "Connection _running is False, resetting", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                    except: pass
                    # #endregion
                    self._conn = None
                # Try a lightweight check to see if connection is alive
                elif hasattr(self._conn, '_connection'):
                    # If underlying connection is closed, reset
                    if self._conn._connection is None:
                        # #region agent log
                        try:
                            with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "Connection _connection is None, resetting", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                        except: pass
                        # #endregion
                        self._conn = None
            except (AttributeError, RuntimeError) as e:
                # #region agent log
                try:
                    with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "Exception during connection health check", "data": {"exception_type": type(e).__name__, "exception_msg": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                except: pass
                # #endregion
                # Connection is in bad state, reset it
                self._conn = None
        
        if self._conn is None:
            # #region agent log
            try:
                with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "Creating new connection", "data": {"db_path": self.db_path}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
            except: pass
            # #endregion
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            # Enable WAL mode for better concurrency
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
        # #region agent log
        try:
            with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "A", "location": "session.py:connect", "message": "connect() returning connection", "data": {"conn_exists": self._conn is not None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
        except: pass
        # #endregion
        return self._conn

    async def execute(self, query: str, params: tuple = ()):
        """
        Execute a query (no commit).
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Cursor result
        """
        conn = await self.connect()
        return await conn.execute(query, params)

    async def commit(self):
        """Commit current transaction."""
        if self._conn:
            await self._conn.commit()

    async def rollback(self):
        """Rollback current transaction."""
        if self._conn:
            await self._conn.rollback()

    async def close(self):
        """Close the connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

