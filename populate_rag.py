
import os
import uuid
import sys
import json
import asyncio
import httpx
import time
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

# Directories to index
# Directories to index
# Production: Load from ENV (JSON list) or default to safe 'data' dir
try:
    env_dirs = os.getenv("RAG_SCAN_DIRS")
    if env_dirs:
        DIRS_TO_SCAN = json.loads(env_dirs)
    else:
        # Default for local dev if not set
        if sys.platform == "linux":
            DIRS_TO_SCAN = ["./data"]
        else:
             # Fallback to current directory for safety if no env var
            DIRS_TO_SCAN = ["./data"] if os.path.exists("./data") else ["."]
except Exception as e:
    print(f"⚠️ Error parsing RAG_SCAN_DIRS: {e}, defaulting to ./data")
    DIRS_TO_SCAN = ["./data"]

# File extensions to include
EXTENSIONS = {'.md', '.txt', '.py', '.json', '.xml', '.html', '.rst', '.ipynb', '.pdf'}

# Semaphore to limit concurrency (Protect GPU RAM)
# 10 is safe for most GPUs; increase to 20-30 if Ollama is fast

CONCURRENCY_LIMIT = int(os.getenv("RAG_CONCURRENCY", 3))

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Simple overlapping chunker (CPU bound)."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ... (Previous helper functions unchanged) ...

async def get_embedding(client: httpx.AsyncClient, text: str, ollama_url: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
    """Generates embedding via Ollama (Async) with Retries."""
    for attempt in range(3):
        try:
            res = await client.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=15.0  # Increased timeout slightly for safety
            )
            if res.status_code == 200:
                return res.json()["embedding"]
            elif res.status_code == 404:
                # Model not found?
                 print(f"❌ Model '{model}' not found in Ollama!")
                 return None
            else:
                if attempt == 2: print(f"❌ Ollama Status {res.status_code}: {res.text[:50]}")
        except Exception as e:
            if attempt == 2: print(f"❌ Connection Error (Final): {e}")
            await asyncio.sleep(1 * (attempt + 1)) # Backoff
    return None


async def process_file(
    sem: asyncio.Semaphore, 
    http_client: httpx.AsyncClient, 
    q_client: AsyncQdrantClient, 
    collection_name: str, 
    fpath: str,
    fname: str,
    ollama_url: str
):
    """Worker function to process a single file."""
    async with sem:
        # 1. Check if already indexed (Optimized Check)
        try:
            # Check if ANY chunk exists for this path
            count_res = await q_client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="path", match=MatchValue(value=fpath))
                    ]
                )
            )
            if count_res.count > 0:
                return "skipped"
        except Exception as e:
            print(f"Warning checking Qdrant: {e}")

        # 2. Read Content
        content = ""
        try:
            ext = os.path.splitext(fname)[1].lower()
            
            # Skip large files (Configurable, default 10MB to avoid OOM)
            max_size = int(os.getenv("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024
            if os.path.exists(fpath):
                fsize = os.path.getsize(fpath)
                if fsize > max_size:
                    return "too_large"

            if ext == '.ipynb':
                content = parse_ipynb(fpath)
            elif ext == '.pdf':
                # Basic PDF support if pypdf installed, else skip
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(fpath)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                except:
                    pass 
            else:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            if len(content) < 50: return "too_small"

        except Exception:
            return "read_error"

        # 3. Chunking (CPU)
        chunks = chunk_text(content)
        if not chunks: return "empty"

        # 4. Embeddings (GPU - Concurrent)
        points = []
        
        # Process chunks in smaller batches if file is huge
        for i, chunk in enumerate(chunks):
            embedding = await get_embedding(http_client, chunk, ollama_url)
            if not embedding: continue
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "filename": fname,
                    "path": fpath,
                    "chunk_index": i,
                    "content": chunk
                }
            ))

    # 5. Upsert (Async)
    if points:
        try:
            await q_client.upsert(
                collection_name=collection_name,
                points=points
            )
            return f"indexed_{len(points)}"
        except Exception as e:
            return "upsert_error"
    
    return "no_embeddings"


async def main():
    print("=== SGR Core: Async RAG Population Script ===")
    print(f"⚙️  Concurrency Limit: {CONCURRENCY_LIMIT}")
    
    # Init Config
    q_host = os.getenv("QDRANT_HOST", "localhost")
    q_port = int(os.getenv("QDRANT_PORT", 6333))
    o_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"Connecting to Qdrant at {q_host}:{q_port}...")
    
    q_client = AsyncQdrantClient(host=q_host, port=q_port)
    collection_name = "finance_docs"
    
    # Check/Create Collection
    try:
        exists = await q_client.collection_exists(collection_name)
        if not exists:
            print(f"Collection '{collection_name}' not found. Creating...")
            await q_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            print(f"Collection '{collection_name}' exists. Appending...")
    except Exception as e:
        print(f"⚠️ Connection Warning: {e}")

    # Gather potential files
    all_files = []
    for root_dir in DIRS_TO_SCAN:
        if not os.path.exists(root_dir):
            print(f"⚠️ Directory not found: {root_dir}")
            continue
            
        print(f"📂 Scanning: {root_dir}")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Optimizations
            dirnames[:] = [d for d in dirnames if d not in {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'site-packages', 'build', 'dist', '.gemini'}]
            
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in EXTENSIONS:
                    all_files.append((os.path.join(dirpath, fname), fname))
    
    print(f"Found {len(all_files)} potential files. Starting async processing...")

    # Runtime
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    start_time = time.time()
    
    async with httpx.AsyncClient() as http_client:
        tasks = []
        for fpath, fname in all_files:
            tasks.append(process_file(sem, http_client, q_client, collection_name, fpath, fname, o_url))
        
        # Run with progress bar logic (simple print)
        total = len(tasks)
        completed = 0
        skipped = 0
        indexed = 0
        
        # Use as_completed for progress updates
        for future in asyncio.as_completed(tasks):
            res = await future
            completed += 1
            
            if res == "skipped":
                skipped += 1
            elif res.startswith("indexed"):
                indexed += 1
                
            # Print progress every 10 files
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed}/{total} | Indexed: {indexed} | Skipped: {skipped} | Speed: {rate:.1f} files/sec", end='\r')
                
    total_time = time.time() - start_time
    print(f"\n✨ Done! Processed {total} files in {total_time:.2f}s. Indexed: {indexed}, Skipped: {skipped}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
