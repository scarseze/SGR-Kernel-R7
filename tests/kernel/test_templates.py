"""KC-I: Template resolution tests — nested fields, missing refs, recursive, json interp."""
import pytest
from tests.kernel.conftest import make_step


class TestTemplateResolution:

    def test_kc_i1_nested_field(self, engine):
        """{{step1.output.a.b}} resolves nested value."""
        outputs = {"step1": {"a": {"b": "deep_value"}}}
        assert engine._resolve_string_template(
            "{{step1.output.a.b}}", outputs) == "deep_value"

    def test_kc_i2_missing_field_safe(self, engine):
        """Missing nested field → placeholder, no crash."""
        outputs = {"step1": {"a": {"b": "val"}}}
        result = engine._resolve_string_template(
            "{{step1.output.a.missing}}", outputs)
        assert "Unresolved" in str(result) or "missing" in str(result)

    def test_kc_i2_missing_step_safe(self, engine):
        """Missing step_id → template stays literal."""
        result = engine._resolve_string_template(
            "{{nonexistent.output}}", {})
        assert "nonexistent" in result

    def test_kc_i3_dict_recursive(self, engine):
        """Templates inside dict values resolve."""
        outputs = {"step1": "resolved_value"}
        result = engine._resolve_params(
            {"key": "{{step1.output}}", "static": "hello"}, outputs)
        assert result["key"] == "resolved_value"
        assert result["static"] == "hello"

    def test_kc_i3_list_recursive(self, engine):
        """Templates inside list items resolve."""
        outputs = {"step1": "val1", "step2": "val2"}
        result = engine._resolve_params(
            {"items": ["{{step1.output}}", "{{step2.output}}", "lit"]}, outputs)
        assert result["items"] == ["val1", "val2", "lit"]

    def test_kc_i1_interpolation_json_dumps(self, engine):
        """Non-primitive in interpolation → json.dumps, not str()."""
        outputs = {"step1": {"key": "val", "num": 42}}
        result = engine._resolve_string_template(
            "Result: {{step1.output}}", outputs)
        assert '"key"' in result
        assert "Result:" in result
