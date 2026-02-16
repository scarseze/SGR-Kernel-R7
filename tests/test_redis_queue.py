import unittest
import os
import asyncio
from unittest.mock import MagicMock, patch
from core.task_queue.redis_queue import RedisTaskQueue
from core.task_queue import TaskStatus

# We need to ensure we can import redis
try:
    import redis.asyncio as redis
    from fakeredis import FakeAsyncRedis
except ImportError:
    redis = None
    FakeAsyncRedis = None

class TestRedisQueue(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if redis is None or FakeAsyncRedis is None:
            self.skipTest("redis or fakeredis not installed")

        # Mock redis.from_url to return FakeAsyncRedis
        self.patcher = patch('redis.asyncio.from_url', side_effect=lambda url, **kwargs: FakeAsyncRedis(decode_responses=True))
        self.mock_from_url = self.patcher.start()
        
        self.queue = RedisTaskQueue("redis://localhost:6379/0")

    async def asyncTearDown(self):
        if hasattr(self, 'patcher'):
            self.patcher.stop()

    async def test_enqueue_and_get(self):
        """Test enqueuing a task and retrieving it."""
        task = await self.queue.enqueue("test_task", {"param": 1})
        self.assertIsNotNone(task.task_id)
        self.assertEqual(task.status, TaskStatus.PENDING)
        
        # Verify in Redis (via get_task)
        fetched = await self.queue.get_task(task.task_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "test_task")
        self.assertEqual(fetched.params["param"], 1)

    async def test_claim_task(self):
        """Test claiming a task from the queue."""
        t1 = await self.queue.enqueue("task_1", {})
        t2 = await self.queue.enqueue("task_2", {})
        
        # Claim first
        claimed = await self.queue.claim_next_task()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task_id, t1.task_id)
        self.assertEqual(claimed.status, TaskStatus.RUNNING)
        self.assertIsNotNone(claimed.started_at)
        
        # Check T1 status update persistence
        fetched_t1 = await self.queue.get_task(t1.task_id)
        self.assertEqual(fetched_t1.status, TaskStatus.RUNNING)
        
        # Claim second
        claimed2 = await self.queue.claim_next_task()
        self.assertEqual(claimed2.task_id, t2.task_id)
        
        # Empty queue
        claimed3 = await self.queue.claim_next_task()
        self.assertIsNone(claimed3)

    async def test_update_status(self):
        """Test updating task status."""
        task = await self.queue.enqueue("test_update", {})
        
        updated = await self.queue.update_status(task.task_id, TaskStatus.COMPLETED, result={"res": 42})
        self.assertEqual(updated.status, TaskStatus.COMPLETED)
        self.assertEqual(updated.result, {"res": 42})
        self.assertIsNotNone(updated.completed_at)
        
        fetched = await self.queue.get_task(task.task_id)
        self.assertEqual(fetched.result, {"res": 42})

if __name__ == '__main__':
    unittest.main()
