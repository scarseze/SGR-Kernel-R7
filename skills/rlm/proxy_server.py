import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Mini-server to run on host and accept LLM requests from Docker container
app = FastAPI()

class LLMQuery(BaseModel):
    prompt: str

@app.post("/llm")
async def query_llm(query: LLMQuery):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured on host")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": query.prompt}],
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            return {"text": response.json()["choices"][0]["message"]["content"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def start_proxy_server(port=8090):
    # Retrieve loop if running in existing async context
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    return server.serve()

if __name__ == "__main__":
    # Standalone run
    uvicorn.run(app, host="0.0.0.0", port=8090)
