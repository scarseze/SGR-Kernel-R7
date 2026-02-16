import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from core.state import Message, AgentState
from core.memory_manager import MemoryManager

class TestMemoryManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_memory = AsyncMock()
        self.mock_summarizer = AsyncMock()
        self.manager = MemoryManager(self.mock_memory, self.mock_summarizer)
        self.manager.summary_trigger_threshold = 5 # Lower threshold for testing
        self.state = AgentState(user_request="test")

    async def test_load_context_simple(self):
        """Test loading context without summary."""
        messages = [Message(role="user", content=f"msg {i}") for i in range(3)]
        self.mock_memory.get_history.return_value = messages
        self.mock_memory.get_last_summary.return_value = None

        await self.manager.load_context("user1", self.state)

        self.assertEqual(self.state.history, messages)
        self.mock_memory.get_history.assert_called()

    async def test_load_context_with_summary(self):
        """Test loading context with existing summary."""
        messages = [Message(role="user", content="recent msg")]
        self.mock_memory.get_history.return_value = messages
        self.mock_memory.get_last_summary.return_value = "Previous summary text"

        await self.manager.load_context("user1", self.state)

        self.assertEqual(len(self.state.history), 2)
        self.assertEqual(self.state.history[0].role, "system")
        self.assertIn("Previous summary text", self.state.history[0].content)
        self.assertEqual(self.state.history[1].content, "recent msg")

    async def test_augment_with_semantic_search(self):
        """Test injecting semantic search results."""
        # Initial state
        self.state.history = [Message(role="user", content="recent", timestamp=datetime.now())]
        
        # Search results
        past_msg = Message(role="user", content="past", timestamp=datetime.now() - timedelta(hours=1))
        self.mock_memory.search_history.return_value = [past_msg]

        await self.manager.augment_with_semantic_search("query", self.state)

        self.assertEqual(len(self.state.history), 2)
        self.assertEqual(self.state.history[0].content, "past")

    async def test_augment_semantic_search_deduplication(self):
        """Test that duplicate messages are not added."""
        t1 = datetime.now()
        msg1 = Message(role="user", content="msg1", timestamp=t1)
        self.state.history = [msg1]
        
        # Search returns same message
        self.mock_memory.search_history.return_value = [msg1]

        await self.manager.augment_with_semantic_search("query", self.state)

        self.assertEqual(len(self.state.history), 1)

    async def test_manage_summarization_trigger(self):
        """Test that summarization is triggered when history exceeds threshold."""
        # Threshold is 5. Return 10 messages.
        messages = [Message(role="user", content=f"msg {i}") for i in range(10)]
        self.mock_memory.get_history.return_value = messages
        self.mock_memory.get_last_summary.return_value = "Old Summary"
        self.mock_summarizer.summarize.return_value = "New Summary"

        await self.manager.manage_summarization("user1")

        # Should fetch history with limit > 5
        self.mock_memory.get_history.assert_called()
        
        # Should call summarizer
        self.mock_summarizer.summarize.assert_called_once()
        
        # Should save summary
        self.mock_memory.save_summary.assert_called_with("user1", "New Summary")

    async def test_manage_summarization_no_trigger(self):
        """Test that summarization is NOT triggered when history is short."""
        # Threshold is 5. Return 3 messages.
        messages = [Message(role="user", content=f"msg {i}") for i in range(3)]
        self.mock_memory.get_history.return_value = messages

        await self.manager.manage_summarization("user1")

        self.mock_summarizer.summarize.assert_not_called()
        self.mock_memory.save_summary.assert_not_called()

if __name__ == '__main__':
    unittest.main()
