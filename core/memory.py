import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, Integer, DateTime, Text, select, desc
from core.state import Message
from core.database import Base, Database, session_context
from core.rag.vector_store import VectorStoreAdapter
from core.rag.embeddings import EmbeddingProvider
import logging

logger = logging.getLogger("core.memory")

class User(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    preferences_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

class Summary(Base):
    __tablename__ = 'summaries'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)

class PersistentMemory:
    def __init__(self, db: Database, vector_store: Optional[VectorStoreAdapter] = None, embedding_provider: Optional[EmbeddingProvider] = None):
        self.db = db
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.collection_name = "episodic_memory"

    async def _get_session(self):
        """Helper to get current request session or create a temporary one."""
        session = session_context.get()
        if session:
            return session
        # Fallback for out-of-request calls (e.g. tests without context)
        # Note: This returns a context manager!
        return self.db.async_session_factory()

    async def add_message(self, user_id: str, role: str, content: str):
        """Save a message to history asynchronously."""
        session = session_context.get()
        if session:
            msg = History(user_id=user_id, role=role, content=content, timestamp=datetime.now())
            # But for streaming updates we might want intermediate commits or flush?
            # Let's flush to get IDs but rely on request scope for commit.
            session.add(msg)
            await session.flush() 
            
            # Index in background (or foreground for now to be safe)
            if self.vector_store and self.embedding_provider:
                try:
                    await self._index_message(user_id, role, content, msg.timestamp)
                except Exception as e:
                    logger.error(f"Failed to index message: {e}")

        else:
            async with self.db.async_session_factory() as session:
                msg = History(user_id=user_id, role=role, content=content, timestamp=datetime.now())
                session.add(msg)
                await session.commit()
                
                if self.vector_store and self.embedding_provider:
                     try:
                        await self._index_message(user_id, role, content, msg.timestamp)
                     except Exception as e:
                        logger.error(f"Failed to index message: {e}")

    async def _index_message(self, user_id: str, role: str, content: str, timestamp: datetime):
        """Create embedding and save to vector store."""
        embedding = await self.embedding_provider.embed(content)
        
        from qdrant_client.models import PointStruct
        
        point_id = str(uuid.uuid4())
        payload = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat(),
            "type": "episodic"
        }
        
        await self.vector_store.upsert(
            collection=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )
        logger.debug(f"Indexed message {point_id} to {self.collection_name}")

    async def get_history(self, user_id: str, limit: int = 20) -> List[Message]:
        """Retrieve recent conversation history."""
        session = session_context.get()
        if session:
            return await self._fetch_history(session, user_id, limit)
        else:
            async with self.db.async_session_factory() as session:
                return await self._fetch_history(session, user_id, limit)

    async def _fetch_history(self, session, user_id, limit):
        stmt = select(History)\
            .where(History.user_id == user_id)\
            .order_by(desc(History.timestamp))\
            .limit(limit)
        
        result = await session.execute(stmt)
        msgs = result.scalars().all()
        
        return [
            Message(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in reversed(msgs)
        ]

    async def search_history(self, query: str, limit: int = 5, score_threshold: float = 0.5) -> List[Message]:
        """Search history using vector similarity."""
        if not self.vector_store or not self.embedding_provider:
            logger.warning("Vector store or embedding provider not configured. Returning empty search.")
            return []
            
        try:
            vector = await self.embedding_provider.embed(query)
            results = await self.vector_store.search(
                collection=self.collection_name,
                vector=vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            messages = []
            for hit in results:
                payload = hit.payload
                # Filter by user_id if needed provided in payload or context? 
                # Ideally we filter in search query but Adapter is generic.
                # Assuming single user or filtering logic later.
                
                messages.append(Message(
                    role=payload.get("role", "unknown"),
                    content=payload.get("content", ""),
                    timestamp=payload.get("timestamp") # Str to datetime conversion needed?
                ))
            return messages
            
        except Exception as e:
            logger.error(f"Search history failed: {e}")
            return []

    async def save_summary(self, user_id: str, content: str):
        """Save a new summary."""
        session = session_context.get()
        if session:
             summary = Summary(user_id=user_id, content=content)
             session.add(summary)
             await session.flush()
        else:
             async with self.db.async_session_factory() as session:
                 summary = Summary(user_id=user_id, content=content)
                 session.add(summary)
                 await session.commit()

    async def get_last_summary(self, user_id: str) -> Optional[str]:
        """Get the most recent summary for the user."""
        stmt = select(Summary).where(Summary.user_id == user_id).order_by(desc(Summary.timestamp)).limit(1)
        
        async def _exec(sess):
            res = await sess.execute(stmt)
            return res.scalar_one_or_none()

        session = session_context.get()
        if session:
            s = await _exec(session)
        else:
            async with self.db.async_session_factory() as session:
                s = await _exec(session)
        
        return s.content if s else None

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences (or create default)."""
        session = session_context.get()
        if session:
            return await self._fetch_prefs(session, user_id)
        else:
            async with self.db.async_session_factory() as session:
                return await self._fetch_prefs(session, user_id)

    async def _fetch_prefs(self, session, user_id):
        stmt = select(User).where(User.user_id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user and user.preferences_json:
            return json.loads(user.preferences_json)
        
        # Default
        # Note: calling update_user_preferences here might recurse if we are not careful,
        # but since we pass session, it should be fine? 
        # Actually update_user_preferences also does session check. 
        # Let's duplicate update logic slightly or call it carefully.
        default_prefs = {"language": "ru"}
        
        # We need to insert default
        prefs_str = json.dumps(default_prefs, ensure_ascii=False)
        new_user = User(user_id=user_id, name="User", preferences_json=prefs_str)
        session.add(new_user)
        # await session.commit() # Don't commit inside getter if managed
        if not session_context.get(): await session.commit()
        
        return default_prefs

    async def update_user_preferences(self, user_id: str, prefs: Dict[str, Any]):
        """Upsert user preferences."""
        session = session_context.get()
        if session:
            await self._upsert_prefs(session, user_id, prefs)
        else:
            async with self.db.async_session_factory() as session:
                await self._upsert_prefs(session, user_id, prefs)
                await session.commit()

    async def _upsert_prefs(self, session, user_id, prefs):
        prefs_str = json.dumps(prefs, ensure_ascii=False)
        stmt = select(User).where(User.user_id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            user.preferences_json = prefs_str
        else:
            user = User(user_id=user_id, name="User", preferences_json=prefs_str)
            session.add(user)
