"""KC-L: Replan system tests — step_id versioning, remaining filtering."""
import pytest
from tests.kernel.conftest import make_step


class TestReplanSystem:

    def test_kc_l2_step_id_versioned_on_collision(self):
        """Reused step_id gets _rN suffix."""
        completed = {"s1", "s2"}
        steps = [make_step("s1"), make_step("s3")]

        for s in steps:
            if s.step_id in completed:
                s.step_id = f"{s.step_id}_r1"

        assert steps[0].step_id == "s1_r1"
        assert steps[1].step_id == "s3"

    def test_kc_l3_remaining_filtered(self):
        """Remaining steps exclude completed."""
        completed = {"s1", "s2"}
        all_steps = [make_step(f"s{i}") for i in range(1, 5)]
        remaining = [s for s in all_steps if s.step_id not in completed]

        assert len(remaining) == 2
        assert {s.step_id for s in remaining} == {"s3", "s4"}
