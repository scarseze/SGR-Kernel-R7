import os
import sys

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Try to load from potential paths
# Ensure we load .env from the script's directory (sgr_core root)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

from core.engine import CoreEngine
from core.logger import configure_logger, get_logger

# Import specific skills
# Import specific skills - REMOVED (using dynamic loader)

import asyncio

logger = get_logger("main")

async def console_approval(msg: str) -> bool:
    """Callback for Human-in-the-Loop approval"""
    print(f"\n⚠️  APPROVAL REQUIRED ⚠️\n{msg}\n")
    try:
        # Run in thread to not block event loop
        user_resp = await asyncio.to_thread(input, "Do you approve this action? [y/N]: ")
        return user_resp.strip().lower() == 'y'
    except EOFError:
        return False

# --- Refactored Engine Builder ---
async def build_engine():
    configure_logger()
    logger.info("Initializing SGR Core...")

    # Configure LLM Strategy
    llm_config = {}
    proxy_url = os.getenv("PROXY_URL")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    if proxy_url:
        logger.info("Using Security Proxy", url=proxy_url)
        llm_config = {
            "base_url": proxy_url + "/v1",
            "api_key": "dummy-key",
            "model": "deepseek-chat"
        }
    elif deepseek_key:
        logger.info("Using Direct DeepSeek API (INSECURE)")
        llm_config = {
            "base_url": "https://api.deepseek.com",
            "api_key": deepseek_key,
            "model": "deepseek-chat"
        }
    elif os.getenv("LLM_BASE_URL"):
        logger.info("Using Custom/Ollama API", url=os.getenv('LLM_BASE_URL'))
        llm_config = {
            "base_url": os.getenv("LLM_BASE_URL"),
            "api_key": os.getenv("LLM_API_KEY", "ollama"),
            "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        }
        
    engine = CoreEngine(llm_config=llm_config, approval_callback=console_approval)
    
    # Dynamic Discovery
    from skills.loader import load_skills
    await load_skills(engine)
    
    return engine

async def main():
    # Force UTF-8 ... (keep existing logic)
    if sys.stdout:
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass 
            
    if sys.stdin:
        try:
            sys.stdin.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass

    # Initialize Engine
    engine = await build_engine()
    
    logger.info("System Ready", status="live")
    print("Ready! System is live. (Type 'exit' to quit)")
    
    while True:
        try:
            # Check for non-interactive handling
            if not sys.stdin.isatty():
                 # Avoid infinite loop in docker logs if no TTY
                 await asyncio.sleep(3600)
                 continue

            # Run blocking input in a separate thread to keep loop responsive
            print("\nYou: ", end="", flush=True)
            try:
                user_input = await asyncio.to_thread(input)
            except EOFError:
                print("EOF Detected. Exiting.")
                break

            if not user_input or not user_input.strip():
                # Ignore empty lines
                continue

            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = await engine.run(user_input)
            print(f"\nAgent:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Critical Error: {e}") 
            await asyncio.sleep(1) # Prevent tight loop on error

import argparse

# ... (Previous Imports) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SGR Core Agent')
    parser.add_argument('--max', action='store_true', help='Run Max Messenger Interface')
    parser.add_argument('--telegram', action='store_true', help='Run Telegram Bot Interface')
    parser.add_argument('--web', action='store_true', help='Run Chainlit Web UI')
    parser.add_argument('--server', action='store_true', help='Run API Server Mode (FastAPI)')
    args = parser.parse_args()

    if args.max:
        from core.interfaces.max_bot import MaxBotInterface
        
        # Init Engine using shared builder
        loop = asyncio.get_event_loop()
        engine = loop.run_until_complete(build_engine())

        # Start Max
        bot = MaxBotInterface(engine)
        try:
            loop.run_until_complete(bot.start())
        except KeyboardInterrupt:
            loop.run_until_complete(bot.stop())

    elif args.server:
        import uvicorn
        from server import app
        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        asyncio.run(main())
