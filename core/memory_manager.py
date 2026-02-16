import logging
from typing import List, Optional, Any, Dict
from core.state import Message, AgentState
from core.memory import PersistentMemory
from core.summarizer import ConversationSummarizer

logger = logging.getLogger("core.memory_manager")

class MemoryManager:
    """
    Orchestrates Short-term (recent history) and Long-term (vector + summary) memory.
    """
    def __init__(self, memory: PersistentMemory, summarizer: ConversationSummarizer):
        self.memory = memory
        self.summarizer = summarizer
        
        # Config
        self.max_recent_messages = 10
        self.summary_trigger_threshold = 20 # If history > 20, trigger summarization

    async def load_context(self, user_id: str, state: AgentState):
        """
        Load context into state:
        1. Recent history (last N)
        2. Relevant episodic memory (RAG) - Handled in Engine run loop currently.
        3. Accumulated Summary
        """
        try:
            # 1. Load Recent
            history = await self.memory.get_history(user_id, limit=self.max_recent_messages)
            
            # 2. Load Summary
            summary_text = await self.memory.get_last_summary(user_id)
            if summary_text:
                summary_msg = Message(role="system", content=f"PREVIOUS CONVERSATION SUMMARY:\n{summary_text}")
                history.insert(0, summary_msg)

            state.history = history
            logger.info("Memory loaded", recent_count=len(history), has_summary=bool(summary_text))
            
            # 3. Check for Summarization Trigger
            # We run this in background ideally, but here we await it for simplicity
            await self.manage_summarization(user_id)
            
        except Exception as e:
            logger.error(f"Failed to load memory context: {e}")
            state.history = []

    async def augment_with_semantic_search(self, query: str, state: AgentState, limit: int = 5):
        """
        Search for relevant past messages and inject them into context.
        """
        try:
            relevant_history = await self.memory.search_history(query, limit=limit)
            if relevant_history:
                existing_timestamps = {m.timestamp for m in state.history}
                found_count = 0
                
                # Prepend returned messages
                for msg in relevant_history:
                    if msg.timestamp not in existing_timestamps:
                        state.history.insert(0, msg)
                        existing_timestamps.add(msg.timestamp)
                        found_count += 1
                
                if found_count > 0:
                    logger.info(f"Augmented context with {found_count} past messages")
                    
        except Exception as e:
            logger.warning(f"Semantic memory search failed: {e}")

    async def manage_summarization(self, user_id: str):
        """
        Check if history is too long and summarize.
        """
        try:
            # Fetch a bit more than threshold to see if we need to summarize
            limit = self.summary_trigger_threshold + 5
            history = await self.memory.get_history(user_id, limit=limit)
            
            if len(history) >= self.summary_trigger_threshold:
                logger.info("Summarization triggered")
                
                # Keep last N messages raw
                keep_count = 5
                if len(history) <= keep_count:
                    return

                msgs_to_compress = history[:-keep_count]
                
                # Get previous summary to merge
                prev_summary = await self.memory.get_last_summary(user_id)
                
                # Create synthetic message for previous summary if exists
                if prev_summary:
                    msgs_to_compress.insert(0, Message(role="system", content=f"Previous Summary: {prev_summary}"))
                
                # Generate new summary
                new_summary = await self.summarizer.summarize(msgs_to_compress)
                
                # Save
                await self.memory.save_summary(user_id, new_summary)
                logger.info("New summary saved")

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
