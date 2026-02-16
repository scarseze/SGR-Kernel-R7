import os
from typing import Type
from datetime import datetime, timedelta
from pydantic import BaseModel

try:
    from ics import Calendar, Event
except ImportError:
    Calendar = None

from skills.base import BaseSkill, SkillMetadata
from skills.calendar.schema import CalendarInput

from core.state import AgentState

class CalendarSkill(BaseSkill):
    name: str = "calendar_skill"
    description: str = "Manages calendar events. Can create .ics files for meetings and reminders."
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["calendar", "event_creation", "ics"],
            risk_level="low",
            side_effects=True,
            idempotent=True,
            requires_network=False,
            requires_filesystem=True,
            cost_class="cheap"
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        return CalendarInput

    async def execute(self, params: CalendarInput, state: AgentState) -> str:
        if not Calendar:
            return "Error: 'ics' library not installed. Please run 'pip install ics'."

        if params.action == 'create_event':
            return self._create_event(params)
        
        return "Unknown action."

    def _create_event(self, params: CalendarInput) -> str:
        try:
            # Parse times
            dt_start = datetime.strptime(params.start_time, "%Y-%m-%d %H:%M:%S")
            
            if params.end_time:
                dt_end = datetime.strptime(params.end_time, "%Y-%m-%d %H:%M:%S")
            else:
                dt_end = dt_start + timedelta(hours=1)

            # Create ICS Event
            c = Calendar()
            e = Event()
            e.name = params.summary
            e.begin = dt_start
            e.end = dt_end
            e.description = params.description or ""
            c.events.add(e)

            # Define Output Path
            # Save to a dedicated 'files' directory in sgr_core root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'generated_files')
            os.makedirs(output_dir, exist_ok=True)

            # Sanitize filename
            safe_summary = "".join([c for c in params.summary if c.isalnum() or c in (' ', '-', '_')]).strip()
            filename = f"{safe_summary}_{dt_start.strftime('%Y%m%d_%H%M')}.ics"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(c.serialize_iter())

            # Return with special marker for Telegram Bot
            return (
                f"📅 **Событие:** {params.summary}\n"
                f"🕒 **Начало:** {dt_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"\n[FILE:{filepath}]"
            )

        except ValueError as e:
            return f"Date format error. Please use YYYY-MM-DD HH:MM:SS. Details: {e}"
        except Exception as e:
            return f"Failed to create event: {e}"
