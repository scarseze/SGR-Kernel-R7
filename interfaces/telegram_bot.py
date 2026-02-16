import os
import asyncio
import logging
import sys

# Add parent dir to sys.path to allow imports from core/skills
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Try loading .env from sgr_core root (parent of interfaces/)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile

from core.engine import CoreEngine
from core.llm import LLMService
from core.voice import VoiceService

# Import Skills
from skills.sglang_sim.handler import SGLangSkill
from skills.portfolio.handler import PortfolioSkill
from skills.xbrl_analyst.handler import XBRLAnalystSkill
from skills.gost_writer.handler import GostWriterSkill
from skills.calendar.handler import CalendarSkill
from skills.logic_rl.handler import LogicRLSkill
from skills.rlm.handler import RLMSkill

# ... (Logging config) ...

# ... (Init logic) ...

# Init LLM Config from env
llm_config = {
    "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    "model": os.getenv("LLM_MODEL", "deepseek-chat")
}

# Initialize Bot and Dispatcher
token = os.getenv("TELEGRAM_TOKEN")
if not token:
    logging.error("TELEGRAM_TOKEN not found in environment variables!")
    # We might want to exit or let it crash, but better to log
    print("❌ Error: TELEGRAM_TOKEN not found in .env")
    sys.exit(1)

bot = Bot(token=token)
dp = Dispatcher()


engine = CoreEngine(llm_config=llm_config)
engine.register_skill(SGLangSkill())
engine.register_skill(PortfolioSkill())
engine.register_skill(XBRLAnalystSkill())
engine.register_skill(GostWriterSkill())
engine.register_skill(CalendarSkill())
engine.register_skill(LogicRLSkill())
engine.register_skill(RLMSkill())

voice_service = VoiceService()

# ... (Start command) ...

async def send_response(message: types.Message, response_text: str):
    """Parses response for [FILE:...] marker and sends file if present."""
    file_path = None
    clean_text = response_text
    
    if "[FILE:" in response_text:
        try:
            # Extract path
            start = response_text.find("[FILE:") + 6
            end = response_text.find("]", start)
            file_path = response_text[start:end].strip()
            
            # Remove marker from text
            clean_text = response_text.replace(f"[FILE:{file_path}]", "").strip()
        except:
            pass

    # Logic: If file exists and text is short, send as caption
    # Otherwise send text separately
    
    if file_path and os.path.exists(file_path):
        try:
            file_input = types.FSInputFile(file_path)
            
            # Use text as caption if it fits and exists
            caption = clean_text if clean_text and len(clean_text) < 1000 else "📂 Ваш файл"
            
            # If we use text as caption, we don't need to send it separately
            if caption == clean_text:
                clean_text = None 
                
            await message.bot.send_document(message.chat.id, file_input, caption=caption, parse_mode="Markdown")
        except Exception as e:
            await message.answer(f"❌ Ошибка отправки файла: {e}")
            
    # Send text if it wasn't used as caption
    if clean_text:
        try:
            await message.answer(clean_text, parse_mode="Markdown")
        except Exception:
            # Fallback
            await message.answer(clean_text)

@dp.message(Command("metrics"))
async def handle_metrics(message: types.Message):
    """Shows RAG metrics for the last request."""
    trace = engine.tracer.get_last_trace()
    if not trace:
        await message.answer("❌ Нет данных о последних запросах.")
        return

    # Basic Info
    txt = f"📊 **Метрики последнего запроса**\n"
    txt += f"🆔 `{trace.request_id[:8]}`\n"
    txt += f"⏱ Полное время: {trace.total_duration:.2f}s\n"
    txt += f"🟢 Статус: {trace.status}\n\n"

    # Plan
    if trace.plan:
         txt += f"📋 **План**: {len(trace.plan.steps)} шагов\n"
         txt += f"🧠 Reasoning: _{trace.plan.reasoning[:100]}..._\n\n"

    # RAG Metrics (Aggregate from all steps)
    rag_queries = []
    for step in trace.steps:
        rag_queries.extend(step.rag_queries)
        
    # Also check planner RAG if any (future compatible)
    
    if not rag_queries:
        txt += "🔍 **RAG**: Не использовался.\n"
    else:
        txt += f"🔍 **RAG ({len(rag_queries)} запросов)**:\n"
        for i, q in enumerate(rag_queries, 1):
            txt += f"\n**#{i}**: `{q.query}`\n"
            if q.rewritten_query:
                txt += f"   🔄 Rewrite: `{q.rewritten_query}`\n"
            txt += f"   ⏱ {q.latency_ms:.0f}ms | 📄 Found: {q.found_docs} | ✅ Used: {q.used_docs}\n"
            txt += f"   📚 Sources: {', '.join([s[:20] for s in q.sources[:3]])}\n"
            
            if q.critique_passed is False:
                 txt += f"   ⚠️ Critique Failed -> Repair: `{q.repair_strategy}`\n"


    await message.answer(txt, parse_mode="Markdown")

@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """
    Handles voice messages:
    1. Downloads the file.
    2. Transcribes via VoiceService (Groq).
    3. Sends text to CoreEngine.
    4. Returns response (Text or File).
    """
    user_id = message.from_user.id
    status_msg = await message.reply("👂 Слушаю...")
    
    file_path = f"temp_voice_{user_id}_{message.voice.file_id}.ogg"
    
    try:
        # 1. Download
        file_id = message.voice.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, file_path)
        
        # 2. Transcribe
        await bot.edit_message_text("✍️ Транскрибирую...", chat_id=message.chat.id, message_id=status_msg.message_id)
        
        # Run transcription in thread to avoid blocking loop
        transcribed_text = await asyncio.to_thread(voice_service.transcribe, file_path)
        
        if not transcribed_text or "Error" in transcribed_text or "Failed" in transcribed_text:
             await bot.edit_message_text(f"❌ Ошибка распознавания: {transcribed_text}", chat_id=message.chat.id, message_id=status_msg.message_id)
             return

        # 3. Process with Agent
        await bot.edit_message_text(f"🧠 Думаю: \"{transcribed_text}\"", chat_id=message.chat.id, message_id=status_msg.message_id)
        
        # Engine run might be slow, so we keep the "Thinking" status or typing action
        await bot.send_chat_action(message.chat.id, action="typing")
        response = await engine.run(transcribed_text)
        
        # Cleanup status message
        await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
        
        # 4. Send Result
        await send_response(message, response)
        
    except Exception as e:
        logging.error(f"Voice Error: {e}")
        await message.answer(f"❌ Ошибка обработки: {e}")
        
    finally:
        # Cleanup file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@dp.message(F.text)
async def handle_text(message: types.Message):
    user_input = message.text
    await bot.send_chat_action(message.chat.id, action="typing")
    
    # engine.run is async, so we just await it
    response = await engine.run(user_input)
    await send_response(message, response)

async def main():
    print("🚀 Telegram Bot Started! (Persona: Buratino)")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped!")
