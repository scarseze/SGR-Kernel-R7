from core.engine import CoreEngine
import os
import time

def test_memory():
    print("=== Testing Memory Persistence ===")
    
    # Ensure env is loaded
    from dotenv import load_dotenv
    # Try loading .env from local directory
    load_dotenv()
    
    # Session 1: Introduce myself
    print("\n--- Session 1 ---")
    
    # Configuration for DeepSeek
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ Error: DEEPSEEK_API_KEY not found in .env")
        return

    deepseek_config = {
        "base_url": "https://api.deepseek.com",
        "api_key": api_key,
        "model": "deepseek-chat"
    }
    
    print(f"Using DeepSeek API connection...")
    
    engine1 = CoreEngine(llm_config=deepseek_config)
    # Mocking user input directly
    q1 = "Hello, my name is Max."
    print(f"User: {q1}")
    resp1 = engine1.run(q1)
    print(f"Agent: {resp1}")
    
    # Simulate partial shutdown
    del engine1
    print("\n... Restarting Agent ...\n")
    
    # Session 2: Ask for name
    print("--- Session 2 ---")
    engine2 = CoreEngine(llm_config=deepseek_config) # New instance, should load DB
    q2 = "What is my name?"
    print(f"User: {q2}")
    resp2 = engine2.run(q2)
    print(f"Agent: {resp2}")
    
    if "Max" in resp2:
        print("\n✅ SUCCESS: Memory works! Agent remembered the name.")
    else:
        print("\n❌ FAILURE: Agent forgot context.")

if __name__ == "__main__":
    # Ensure env is loaded (hack for test)
    from dotenv import load_dotenv
    load_dotenv()
    test_memory()
