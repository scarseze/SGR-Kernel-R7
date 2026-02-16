import sys
import os
import requests
import json
import argparse

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

def read_file_context(file_path: str, start_line: int = None, end_line: int = None):
    """
    Simulates reading a file from an IDE.
    """
    abs_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        return None

    with open(abs_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        content = "".join(lines)
    
    selection = None
    if start_line and end_line:
        # subset lines (1-based to 0-based)
        sel_lines = lines[start_line-1:end_line]
        selection = {
            "start_line": start_line,
            "end_line": end_line,
            "text": "".join(sel_lines)
        }
        print(f"📝 Selected lines {start_line}-{end_line} ({len(sel_lines)} lines)")

    return {
        "file_path": abs_path,
        "content": content,
        "selection": selection,
        "cursor_line": start_line if start_line else 1
    }

def main():
    parser = argparse.ArgumentParser(description="Mock Editor acting as an Inverse MCP Client")
    parser.add_argument("--file", type=str, required=True, help="File to 'edit'")
    parser.add_argument("--query", type=str, required=True, help="Request to the agent")
    parser.add_argument("--lines", type=str, help="Line range e.g. 10-20")
    
    args = parser.parse_args()
    
    start_line, end_line = None, None
    if args.lines:
        try:
            parts = args.lines.split('-')
            start_line = int(parts[0])
            end_line = int(parts[1])
        except:
            print("❌ Invalid format for lines. Use start-end (e.g. 10-20)")
            return

    # 1. Gather Context
    print(f"📂 Opening file: {args.file}")
    context = read_file_context(args.file, start_line, end_line)
    if not context:
        return

    # 2. Payload Construction
    payload = {
        "query": args.query,
        "source_app": "mock_editor_v1",
        "context": context
    }
    
    # 3. Send to Agent
    url = "http://127.0.0.1:8000/v1/agent/process"
    print(f"🚀 Sending context to Agent at {url}...")
    
    try:
        # Use session to ignore proxies (important for localhost)
        session = requests.Session()
        session.trust_env = False
        
        response = session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("\n🤖 --- Agent Response --- 🤖\n")
        print(result["result"])
        print("\n--------------------------")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Agent. Is server.py running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
