from typing import List
from core.state import Message
from core.llm import LLMService
import logging

logger = logging.getLogger("core.memory.summarizer")

class ConversationSummarizer:
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def summarize(self, messages: List[Message]) -> str:
        """
        Summarize a list of messages into a concise paragraph.
        """
        if not messages:
            return ""

        conversation_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        prompt = f"""
        Please summarize the following conversation history into a concise but informative paragraph.
        Focus on key facts, user preferences, and important decisions made.
        Do not lose critical details.
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        try:
            summary = await self.llm.generate(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return "Error generating summary."
