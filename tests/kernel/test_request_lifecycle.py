"""KC-A: Request Lifecycle tests."""
import pytest
from core.security import SecurityViolationError
from tests.kernel.conftest import make_skill


class TestRequestLifecycle:

    @pytest.mark.asyncio
    async def test_kc_a1_security_input_gate(self, engine):
        """KC-A1: Malicious input blocked BEFORE any skill runs."""
        skill = make_skill()
        engine.skills["test_skill"] = skill

        with pytest.raises(SecurityViolationError):
            engine.security.validate("nc -l -p 4444")

        skill.execute.assert_not_called()
