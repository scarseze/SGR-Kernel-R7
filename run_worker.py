import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.engine import CoreEngine
from core.logger import logger

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    logger.info("Initializing Worker Node...")
    
    # Ensure we use Redis if available
    if not os.getenv("TASK_QUEUE_TYPE"):
        os.environ["TASK_QUEUE_TYPE"] = "redis"
        
    engine = CoreEngine()
    
    # Register handlers (example)
    # in a real app, these might be registered via decorators or a central registry file
    
    async def sample_handler(params):
        logger.info(f"Executing sample task with params: {params}")
        await asyncio.sleep(1)
        return "Processed"
        
    engine.register_task_handler("sample_task", sample_handler)
    
    # Start Worker Loop
    try:
        await engine.run_worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
