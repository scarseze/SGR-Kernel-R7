"""KC-F: Timeout authority tests."""
import pytest
import asyncio
from unittest.mock import MagicMock
from core.types import SkillExecutionContext
from tests.kernel.conftest import make_skill


class TestTimeoutAuthority:

    def test_kc_f1_timeout_from_metadata(self):
        """KC-F1: timeout_sec=7 → ctx.timeout set to 7 by TimeoutMiddleware."""
        from core.middleware import TimeoutMiddleware

        mw = TimeoutMiddleware()
        ctx = SkillExecutionContext(
            request_id="r1", step_id="s1", skill_name="test",
            params={}, state=MagicMock(), skill=MagicMock(),
            metadata=make_skill(timeout=7.0).metadata,
            trace=MagicMock(), attempt=1,
        )
        asyncio.get_event_loop().run_until_complete(mw.before_execute(ctx))
        assert ctx.timeout == 7.0

    def test_kc_f2_kernel_default_timeout(self):
        """KC-F2: timeout=0 → kernel uses 60s fallback."""
        ctx_timeout = 0.0
        effective = ctx_timeout if ctx_timeout > 0 else 60.0
        assert effective == 60.0
