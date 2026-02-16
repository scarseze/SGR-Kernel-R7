import unittest
from unittest.mock import MagicMock, patch
import os

# Set dummy env var to trigger telemetry init
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

from core.telemetry import TelemetryManager, init_telemetry, get_telemetry

class TestTelemetry(unittest.TestCase):
    def setUp(self):
        # Reset global instance
        import core.telemetry
        core.telemetry._telemetry_instance = None

    def test_init_success(self):
        """Test that telemetry initializes when packages are present."""
        # We know packages are installed in this env
        tm = init_telemetry()
        self.assertTrue(tm.enabled)
        self.assertIsNotNone(tm.tracer)
        
        with tm.span("test_span") as span:
            # Span might be None if failed, but if enabled it should be opaque object
            pass

    def test_singleton(self):
        # Reset first to ensure clean state
        import core.telemetry
        core.telemetry._telemetry_instance = None
        
        t1 = init_telemetry()
        t2 = init_telemetry() 
        t3 = get_telemetry()
        
        self.assertIs(t1, t2)
        self.assertIs(t1, t3)

if __name__ == '__main__':
    unittest.main()
