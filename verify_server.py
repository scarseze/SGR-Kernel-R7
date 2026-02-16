import subprocess
import time
import requests
import sys
import os

def test_server():
    print("🚀 Starting SGR Core Server...")
    
    # Force unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        [sys.executable, "main.py", "--server"],
        cwd=r"C:\Users\macht\SA\sgr_core",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    try:
        print("⏳ Waiting for server to initialize (30s)...")
        # Monitor stdout for readiness (optional improvement)
        time.sleep(30)
        
        url = "http://127.0.0.1:8000/v1/agent/process"
        payload = {
            "query": "Research deep learning trends 2025",
            "source_app": "verification_script",
            "context": {"user": "test_user"}
        }
        
        # Use session with trust_env=False to ignore proxies
        session = requests.Session()
        session.trust_env = False
        
        print(f"🔌 Sending request to {url} (ignoring proxies)...")
        try:
            response = session.post(url, json=payload, timeout=30)
            print(f"📥 Status Code: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
            if response.status_code == 200:
                print("\n✅ VERIFICATION SUCCESSFUL!")
            else:
                print("\n❌ VERIFICATION FAILED: Non-200 response")
                # Print server logs
                stdout, stderr = process.communicate(timeout=5)
                print("--- SERVER LOGS START ---")
                print(stdout)
                print("--- SERVER LOGS END ---")
                print("--- SERVER ERRORS START ---")
                print(stderr)
                print("--- SERVER ERRORS END ---")
                
        except requests.exceptions.RequestException as e:
            print(f"\n❌ VERIFICATION FAILED: Request Error: {e}")
            stdout, stderr = process.communicate(timeout=5)
            print("--- SERVER LOGS START ---")
            print(stdout)
            print("--- SERVER LOGS END ---")
            print("--- SERVER ERRORS START ---")
            print(stderr)
            print("--- SERVER ERRORS END ---")

    finally:
        print("🛑 Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    test_server()
