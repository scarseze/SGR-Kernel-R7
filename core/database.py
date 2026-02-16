import os
import contextlib
from contextvars import ContextVar
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Global Context for Session
session_context: ContextVar[AsyncSession] = ContextVar("session_context", default=None)

class Database:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./memory.db")
        
        # Ensure async driver
        if "sqlite" in self.db_url and "+aiosqlite" not in self.db_url:
             self.db_url = self.db_url.replace("sqlite://", "sqlite+aiosqlite://")

        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {}
        )
        
        self.async_session_factory = async_sessionmaker(
            self.engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )

    async def init_db(self):
        """Create tables asynchronously."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @contextlib.asynccontextmanager
    async def session(self):
        """
        Request-scope session context manager.
        Sets the global ContextVar for the duration of the block.
        """
        session = self.async_session_factory()
        token = session_context.set(session)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            session_context.reset(token)
