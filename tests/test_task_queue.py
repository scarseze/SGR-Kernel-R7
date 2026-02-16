import pytest
import asyncio
from core.task_queue import TaskStatus

@pytest.mark.asyncio
async def test_enqueue_task(engine):
    task = await engine.task_queue.enqueue("test_task", {"a": 1})
    assert task.task_id is not None
    assert task.status == TaskStatus.PENDING
    assert task.params == {"a": 1}

    # Verify DB
    saved_task = await engine.task_queue.get_task(task.task_id)
    assert saved_task is not None
    assert saved_task.task_id == task.task_id
    assert saved_task.name == "test_task"

@pytest.mark.asyncio
async def test_worker_processing(engine):
    # Setup handler
    handler_called = asyncio.Event()
    received_params = {}

    async def task_handler(params):
        received_params.update(params)
        handler_called.set()
        return "success"

    engine.register_task_handler("worker_task", task_handler)

    # Submit task
    task_id = await engine.submit_task("worker_task", {"key": "value"})

    # Run worker for a short time
    # We can use asyncio.wait_for or just let it run in background and check status
    worker_task = asyncio.create_task(engine.run_worker(interval=0.01))

    try:
        await asyncio.wait_for(handler_called.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pass
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    assert handler_called.is_set()
    assert received_params == {"key": "value"}
    
    # Check status
    task = await engine.task_queue.get_task(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "success"

@pytest.mark.asyncio
async def test_claim_order(engine):
    # Depending on test order, DB might have dirty state if shared.
    # The fixture uses "sqlite:///:memory:" which is usually per connection, 
    # but SQLAlchemy with async engine might hold it differently.
    # We should clean up or assume fresh.
    # Assuming fresh from fixture for now given it's function scoped usually (pytest default, but check conftest)
    # The conftest says @pytest.fixture which is function scoped.
    
    await engine.task_queue.enqueue("task1", {})
    await engine.task_queue.enqueue("task2", {})
    
    t1 = await engine.task_queue.claim_next_task()
    assert t1.name == "task1"
    assert t1.status == TaskStatus.RUNNING
    
    t2 = await engine.task_queue.claim_next_task()
    assert t2.name == "task2"
    assert t2.status == TaskStatus.RUNNING
    
    t3 = await engine.task_queue.claim_next_task()
    assert t3 is None
