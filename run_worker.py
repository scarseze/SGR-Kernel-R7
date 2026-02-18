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
    # Create Engine with Worker Mode config
    # Workers typically don't need the full LLM config if they just execute tools, 
    # but for Agent Workers, they might.
    
    # In V1.x, the CoreEngine handles task polling internally if configured as a worker.
    # The 'run_worker' method is the entry point.
    
    try:
        engine = CoreEngine()
        logger.info("Worker Node Started. Polling for tasks...")
        await engine.run_worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    except Exception as e:
        logger.error(f"Worker crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
