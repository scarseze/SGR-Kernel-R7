from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING
from core.types import Capability, RiskLevel, CostClass, SkillMetadata

if TYPE_CHECKING:
    from core.llm import LLMService, ModelPool

class ModelTier(str, Enum):
    FAST = "fast"      # e.g., gpt-3.5-turbo, deepseek-coder-1.3b
    MID = "mid"        # e.g., gpt-4o-mini, deepseek-coder-6.7b
    HEAVY = "heavy"    # e.g., gpt-4-turbo, deepseek-coder-33b

class TierRouter:
    """
    Intelligent router to select the appropriate model tier based on task characteristics.
    """
    
    def __init__(self, model_pool: 'ModelPool'):
        self.pool = model_pool

    def route(self, metadata: SkillMetadata, schema: Optional[Dict[str, Any]] = None, attempt: int = 1) -> 'LLMService':
        """
        Selects the best LLM service based on skill metadata, schema complexity, and attempt number.
        """
        tier = self._determine_tier(metadata, schema, attempt)
        return self.pool.get(tier)

    def _determine_tier(self, metadata: SkillMetadata, schema: Optional[Dict[str, Any]], attempt: int) -> ModelTier:
        # 0. Risk + Approval -> Tier Lock (Rule 7)
        # If High Risk AND requires approval -> ALWAYS HEAVY
        if metadata.risk_level == RiskLevel.HIGH and metadata.requires_approval_hint:
             return ModelTier.HEAVY

        # 1. Compute base tier from capabilities, risk, schema
        base_tier = self._get_base_tier(metadata, schema)
        
        # 2. Escalation (Rule 1)
        # attempt 1 -> base logic
        # attempt 2 -> bump tier
        # attempt 3+ -> bump tier again
        if attempt > 1:
            base_tier = self._escalate_tier(base_tier, attempt)
            
        # 3. Tier Floor/Ceiling (Rule 6)
        if metadata.min_tier:
            base_tier = self._apply_floor(base_tier, metadata.min_tier)
        if metadata.max_tier:
            base_tier = self._apply_ceiling(base_tier, metadata.max_tier)
            
        return base_tier

    def _apply_floor(self, current: ModelTier, floor: str) -> ModelTier:
        try:
            floor_tier = ModelTier(floor)
        except ValueError:
            return current # Invalid floor ignored
            
        # Order: fast < mid < heavy
        order = {ModelTier.FAST: 1, ModelTier.MID: 2, ModelTier.HEAVY: 3}
        if order.get(current, 0) < order.get(floor_tier, 0):
            return floor_tier
        return current

    def _apply_ceiling(self, current: ModelTier, ceiling: str) -> ModelTier:
        try:
            ceiling_tier = ModelTier(ceiling)
        except ValueError:
            return current
            
        order = {ModelTier.FAST: 1, ModelTier.MID: 2, ModelTier.HEAVY: 3}
        if order.get(current, 0) > order.get(ceiling_tier, 0):
            return ceiling_tier
        return current

    def _get_base_tier(self, metadata: SkillMetadata, schema: Optional[Dict[str, Any]]) -> ModelTier:
        # 1. High Risk -> HEAVY
        if metadata.risk_level == RiskLevel.HIGH:
            return ModelTier.HEAVY
            
        # 2. Complex Capabilities -> HEAVY
        if Capability.REASONING in metadata.capabilities or Capability.PLANNING in metadata.capabilities:
            return ModelTier.HEAVY
            
        # 3. Code Generation -> MID (usually needs better context than fast)
        if Capability.CODE in metadata.capabilities:
            return ModelTier.MID

        # 4. Schema Complexity (Rule 2)
        if schema:
            complexity = self._calculate_schema_complexity(schema)
            if complexity > 10: # Arbitrary threshold for "complex" schema
                return ModelTier.MID
                
        # 5. Default -> FAST
        return ModelTier.FAST
        
    def _escalate_tier(self, current_tier: ModelTier, attempt: int) -> ModelTier:
        """
        Escalation logic:
        Fast -> Mid (attempt 2)
        Mid -> Heavy (attempt 3+)
        Heavy -> Heavy
        """
        # If attempt is high, force upgrade
        if attempt == 2:
            if current_tier == ModelTier.FAST: return ModelTier.MID
        elif attempt >= 3:
            if current_tier == ModelTier.FAST: return ModelTier.HEAVY
            if current_tier == ModelTier.MID: return ModelTier.HEAVY
            
        return current_tier

    def _calculate_schema_complexity(self, schema: Dict[str, Any]) -> int:
        """
        Simple heuristic: count nodes in JSON schema.
        """
        count = 0
        if isinstance(schema, dict):
            count += 1
            for v in schema.values():
                count += self._calculate_schema_complexity(v)
        elif isinstance(schema, list):
            for v in schema:
                count += self._calculate_schema_complexity(v)
        return count
