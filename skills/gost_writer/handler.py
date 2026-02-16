import os
from typing import Type
from pydantic import BaseModel
try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None

from core.state import AgentState
from core.llm import LLMService
from skills.base import BaseSkill, SkillMetadata
from skills.gost_writer.schema import GostWriterInput, GostData

class GostWriterSkill(BaseSkill):
    name: str = "gost_writer"
    description: str = (
        "Generates formal technical documentation in GOST 34 style (TZ, Manual, PMI). "
        "Uses AI to draft the content based on a brief topic."
    )
    _llm: LLMService = None

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["technical_writing", "gost", "documentation"],
            risk_level="high",  # Generates complex doc, cost
            side_effects=False, # Returns text
            idempotent=False,
            requires_network=True,
            requires_filesystem=True,
            cost_class="expensive"
        )

    def __init__(self, llm_service: LLMService = None, **data):
        super().__init__(**data)
        self._llm = llm_service or LLMService()

    @property
    def input_schema(self) -> Type[BaseModel]:
        return GostWriterInput
        
    def is_sensitive(self, params: GostWriterInput) -> bool:
        # Generating a GOST document is expensive/complex, so we ask for approval
        return True

    async def execute(self, params: GostWriterInput, state: AgentState) -> str:
        if Environment is None:
            return "Error: 'jinja2' library is missing."

        if params.action == "generate":
            if params.template_type != "tz":
                return f"Error: Currently only 'tz' template is fully migrated. You asked for {params.template_type}."

            # 1. Retrieve Context from RAG (if available)
            rag_context = ""
            if hasattr(self, 'rag') and self.rag:
                print(f"[{self.name}] Searching RAG for similar documents...")
                # Search for similar TZs or standards
                # Using 'finance_docs' for now as it's the only collection, but ideally should be separate
                results = self.rag.search(collection_name="finance_docs", query=params.topic, limit=2)
                if results and results[0]['score'] > 0.0: # Check if result is valid
                    docs = [r['content'][:500] + "..." for r in results]
                    rag_context = "\nReferences from Knowledge Base:\n" + "\n\n".join(docs)

            # 2. Generate Data using LLM
            print(f"[{self.name}] Generating GOST data for topic: '{params.topic}'...")
            try:
                gost_data = await self._llm.generate_structured(
                    system_prompt=(
                        "You are an expert Systems Analyst writing a Russian GOST 34 Technical Specification (TZ).\n"
                        "Fill out the schema based on the user's description.\n"
                        "Use formal bureaucratic Russian language.\n"
                        "Be detailed and realistic.\n"
                        f"{rag_context}"
                    ),
                    user_prompt=f"System Description: {params.topic}",
                    response_model=GostData
                )
            except Exception as e:
                return f"Failed to generate document data: {e}"

            # 2. Render Template
            try:
                # Path to templates relative to this file
                base_dir = os.path.dirname(os.path.abspath(__file__))
                template_dir = os.path.join(base_dir, 'templates')
                
                env = Environment(loader=FileSystemLoader(template_dir))
                template_name = "gost_34_tz.md" # Fixed for now as per schema logic
                template = env.get_template(template_name)
                
                rendered_content = template.render(gost_data.model_dump())
                
                # Optional: Save to file implicitly or just return text
                # We return text so the user (or Core Agent) can decide what to do
                return f"### Generated GOST 34 TZ for {params.topic}\n\n{rendered_content}"
                
            except Exception as e:
                return f"Template Rendering Failed: {e}"

        return "Unknown action."
