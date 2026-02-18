"""
Skill Capability Registry for SGR Kernel.
"""
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

class Capability(BaseModel):
    """
    Definition of a capability.
    """
    name: str # e.g. "read_fs", "write_fs", "network", "compute"
    description: str
    risk_level: str = "low"

class SkillMethod(BaseModel):
    """
    Metadata about a specific skill method.
    """
    name: str
    required_capabilities: List[str] = Field(default_factory=list)
    description: str
    side_effects: bool = False
    
class CapabilityRegistry:
    """
    Registry of all available capabilities and skill mappings.
    """
    _capabilities: Dict[str, Capability] = {}
    _skill_map: Dict[str, SkillMethod] = {}
    
    @classmethod
    def register_capability(cls, cap: Capability):
        cls._capabilities[cap.name] = cap
        
    @classmethod
    def register_skill(cls, skill_name: str, method: SkillMethod):
        # Validate capabilities exist
        for cap_name in method.required_capabilities:
            if cap_name not in cls._capabilities:
                raise ValueError(f"Capability {cap_name} not registered.")
                
        cls._skill_map[skill_name] = method
        
    @classmethod
    def get_skill(cls, skill_name: str) -> Optional[SkillMethod]:
        return cls._skill_map.get(skill_name)
    
    @classmethod
    def check_capabilities(
        cls, 
        skill_name: str, 
        granted_capabilities: Set[str]
    ) -> bool:
        """
        Check if the granted capabilities are sufficient for the skill.
        """
        skill = cls.get_skill(skill_name)
        if not skill:
            return False # Unknown skill is unsafe
            
        required = set(skill.required_capabilities)
        return required.issubset(granted_capabilities)

# Register default capabilities
CapabilityRegistry.register_capability(Capability(name="read_fs", description="Read from file system"))
CapabilityRegistry.register_capability(Capability(name="write_fs", description="Write to file system", risk_level="high"))
CapabilityRegistry.register_capability(Capability(name="network", description="Network access", risk_level="medium"))
CapabilityRegistry.register_capability(Capability(name="compute", description="Heavy compute", risk_level="low"))
CapabilityRegistry.register_capability(Capability(name="llm_heavy", description="LLM heavy reasoning", risk_level="medium"))
CapabilityRegistry.register_capability(Capability(name="external_api", description="Call external APIs", risk_level="medium"))
