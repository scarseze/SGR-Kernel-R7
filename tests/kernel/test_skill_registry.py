"""KC-M: Skill registration tests — capability index, metadata immutability."""
import pytest
from unittest.mock import MagicMock
from core.types import SkillMetadata, Capability
from tests.kernel.conftest import make_skill


class TestSkillRegistration:

    def test_kc_m1_capability_index_built(self, engine):
        """Registration populates cap_index."""
        skill = make_skill(capabilities=[Capability.REASONING, Capability.WEB])
        engine.register_skill(skill)
        assert "test_skill" in engine.skills
        assert Capability.REASONING in engine.cap_index or \
               "reasoning" in str(engine.cap_index)

    def test_kc_m2_dict_metadata_normalized(self, engine):
        """Dict metadata → SkillMetadata after registration."""
        skill = MagicMock()
        skill.name = "dict_meta_skill"
        skill.metadata = {
            "name": "dict_meta_skill",
            "capabilities": ["reasoning"],
            "description": "test",
        }
        engine.register_skill(skill)
        assert isinstance(engine.skills["dict_meta_skill"].metadata, SkillMetadata)
        assert engine.skills["dict_meta_skill"].metadata.name == "dict_meta_skill"

    def test_kc_m2_estimated_cost_on_model(self):
        """estimated_cost is a proper model field."""
        meta = SkillMetadata(
            name="x", capabilities=[Capability.REASONING],
            estimated_cost=1.5
        )
        assert meta.estimated_cost == 1.5
        meta2 = SkillMetadata(name="y", capabilities=[Capability.REASONING])
        assert meta2.estimated_cost == 0.0
