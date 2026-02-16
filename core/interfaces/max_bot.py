"""
SGR Core - Max Messenger Interface
Adapts the "Max" Bot API (REST) to SGR Core's Logic Engine.
"""

import os
import asyncio
import logging
import httpx
from typing import Optional, Dict, Any, List
from core.engine import CoreEngine
from core.models import UserMessage

logger = logging.getLogger(__name__)

class MaxBotInterface:
    def __init__(self, engine: CoreEngine):
        self.engine = engine
        self.token = os.getenv("MAX_BOT_TOKEN")
        # Default API endpoint (placeholder, update with real one from docs)
        self.api_url = os.getenv("MAX_API_URL", "https://api.max.ru/bot/v1")
        
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = False
        self.offset = 0  # For polling

    async def start(self):
        """Starts the Max Bot polling loop."""
        if not self.token:
            logger.warning("⚠️ MAX_BOT_TOKEN not found. Max Interface disabled.")
            return

        self.is_running = True
        logger.info(f"🚀 Max Bot Interface started (Endpoint: {self.api_url})")

        while self.is_running:
            try:
                updates = await self.get_updates()
                for update in updates:
                    await self.process_update(update)
            except Exception as e:
                logger.error(f"Error in Max polling loop: {e}")
                await asyncio.sleep(5)  # Backoff on error
            
            await asyncio.sleep(1.0) # Polling interval

    async def stop(self):
        """Stops the polling loop."""
        self.is_running = False
        await self.client.aclose()
        logger.info("🛑 Max Bot Interface stopped.")

    async def get_updates(self) -> List[Dict]:
        """Fetches new messages from Max API (Long Polling simulation)."""
        # Note: This is a hypothetical API structure based on standard Bot API patterns
        # Real MAX API might differ slightly (e.g. /events or /updates)
        url = f"{self.api_url}/updates?token={self.token}&offset={self.offset}"
        
        try:
            resp = await self.client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if not data: return []
                
                # Assume standard structure: {"result": [...], "ok": true}
                items = data.get("result", [])
                if items:
                    # Update offset to last_id + 1
                    self.offset = items[-1].get("update_id", self.offset) + 1
                return items
            else:
                logger.debug(f"Max API Fetch Failed: {resp.status_code} - {resp.text}")
                return []
        except Exception:
            # logger.warning(f"Connection error to Max API: {e}")
            return []

    async def process_update(self, update: Dict):
        """Converts Max Update -> SGR UserMessage -> Core Engine."""
        # Extract message data (Hypothetical structure)
        message = update.get("message", {})
        if not message: return

        text = message.get("text")
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")

        if not text or not chat_id: return

        # Log incoming
        logger.info(f"📩 [MAX] New message from {user_id}: {text[:50]}...")

        # 1. Adapt to Internal Schema
        user_msg = UserMessage(
            text=text,
            user_id=str(user_id),
            platform="max",
            meta={"chat_id": chat_id}
        )

        # 2. Send to Brain (Core Engine)
        response_text = await self.engine.process_message(user_msg)

        # 3. Send Answer back to Max
        await self.send_message(chat_id, response_text)

    async def send_message(self, chat_id: str, text: str):
        """Sends a text message back to the user via Max API."""
        url = f"{self.api_url}/messages/send"
        payload = {
            "token": self.token,
            "chat_id": chat_id,
            "text": text,
            # "parse_mode": "Markdown" # Assuming Markdown support
        }
        
        try:
            resp = await self.client.post(url, json=payload)
            if resp.status_code != 200:
                logger.error(f"❌ Failed to send Max message: {resp.text}")
        except Exception as e:
            logger.error(f"❌ Network error sending Max message: {e}")

