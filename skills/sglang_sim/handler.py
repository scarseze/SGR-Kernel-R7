from typing import Type, List
from pydantic import BaseModel, Field

from core.state import AgentState
from core.llm import LLMService
from skills.base import BaseSkill, SkillMetadata
from skills.sglang_sim.schema import RiskAnalysisInput, ClientData

class CreditRiskResponse(BaseModel):
    risk_score: int
    decision: str
    reason: str

class SGLangSkill(BaseSkill):
    name: str = "credit_risk_analyst"
    description: str = "Analyzes credit risk for a list of clients using a bank simulation model."

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            capabilities=["financial_analysis", "credit_risk"],
            risk_level="low",
            side_effects=False,
            idempotent=False,
            requires_network=True,
            requires_filesystem=False,
            cost_class="medium"
        )

    def __init__(self, llm_service: LLMService = None):
        # Use provided LLM or create a default one
        self._llm = llm_service or LLMService()

    @property
    def input_schema(self) -> Type[BaseModel]:
        return RiskAnalysisInput

    async def execute(self, params: RiskAnalysisInput, state: AgentState) -> str:
        results = []
        print(f"[{self.name}] Processing {len(params.clients)} clients...")

        for client in params.clients:
            # Reusing the Core LLM Service for the inner logic
            # mimicking the SGLang 'program'
            system_prompt = (
                "You are a bank risk assessment API.\n"
                "Output ONLY valid JSON."
            )
            user_prompt = (
                f"Client Data:\n"
                f"Name: {client.name}\n"
                f"Salary: {client.salary}\n"
                f"Credit History: {client.credit_history}"
            )
            
            try:
                decision = await self._llm.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=CreditRiskResponse
                )
                icon = "✅" if decision.decision == "APPROVE" else "❌"
                results.append(f"{icon} **{client.name}**: {decision.decision} (Score: {decision.risk_score})\n   Reason: {decision.reason}")
            except Exception as e:
                results.append(f"⚠️ Error processing {client.name}: {e}")

        summary = "\n\n".join(results)
        return f"### Credit Risk Analysis Report\n\n{summary}"
