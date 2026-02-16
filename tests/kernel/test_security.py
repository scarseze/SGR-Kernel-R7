"""KC-G: Security enforcement tests — param validation, output sanitization."""
import pytest
from core.security import SecurityGuardian, SecurityViolationError


class TestSecurityEnforcement:

    def test_kc_g1_param_injection(self):
        """KC-G1: Dangerous value in resolved params → blocked."""
        sec = SecurityGuardian()
        with pytest.raises(SecurityViolationError):
            sec.validate_params({"cmd": "nc -l -p 4444", "safe": "hello"})

    def test_kc_g1_nested_injection(self):
        """KC-G1: Nested params with dangerous values also caught."""
        sec = SecurityGuardian()
        with pytest.raises(SecurityViolationError):
            sec.validate_params({"outer": {"inner": "rm -rf /"}})

    def test_kc_g2_output_api_key_leak(self):
        """KC-G2: Leaked API key → blocked."""
        sec = SecurityGuardian()
        with pytest.raises(SecurityViolationError):
            sec.validate_output("result: api_key = sk-12345abc")

    def test_kc_g2_output_private_key(self):
        """KC-G2: Private key in output → blocked."""
        sec = SecurityGuardian()
        with pytest.raises(SecurityViolationError):
            sec.validate_output("-----BEGIN RSA PRIVATE KEY-----\nMIIBog...")

    def test_kc_g2_clean_output_passes(self):
        """KC-G2: Normal output passes validation."""
        sec = SecurityGuardian()
        sec.validate_output("Revenue increased by 15%.")
