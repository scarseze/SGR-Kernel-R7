import os
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger("core.telemetry")

# Global Telemetry Instance
_telemetry_instance = None

class TelemetryManager:
    def __init__(self, service_name: str = "sgr-core", endpoint: str = "http://jaeger:4317"):
        self.enabled = False
        self.tracer = None
        self.meter = None

        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter

            resource = Resource.create({"service.name": service_name})

            # 1. Tracing
            trace_provider = TracerProvider(resource=resource)
            if endpoint:
                try:
                    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                    span_processor = BatchSpanProcessor(otlp_exporter)
                    trace_provider.add_span_processor(span_processor)
                    logger.info(f"OpenTelemetry Tracing enabled (endpoint: {endpoint})")
                except Exception as e:
                    logger.warning(f"Failed to configure OTLP exporter: {e}")
            
            # Check if provider already set to avoid error in tests/re-init
            # trace.get_tracer_provider() returns a Proxy or NoOp by default. 
            # We just set it. If it fails, we catch it? No, otel warns usually.
            # But let's only set if it's not already ours.
            try:
                trace.set_tracer_provider(trace_provider)
            except Exception:
                 # Provider might be already set (e.g. during tests)
                 pass
                 
            self.tracer = trace.get_tracer(service_name)

            # 2. Metrics (Placeholder for now, or Prometheus if library allows)
            # For now standard print or no-op
            self.meter = metrics.get_meter(service_name)
            
            self.enabled = True

        except ImportError:
            logger.warning("OpenTelemetry packages not found. Telemetry disabled. Install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Telemetry: {e}")
            self.enabled = False

    @contextmanager
    def span(self, name: str, attributes: dict = None):
        """Context manager for creating a span."""
        if not self.enabled or not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span

    def log_metric(self, name: str, value: float, attributes: dict = None):
        """Log a metric (Counter/Gauge placeholder)."""
        if not self.enabled:
            return
        # TODO: Implement proper metric recording
        pass

def init_telemetry(service_name: str = "sgr-core"):
    global _telemetry_instance
    if _telemetry_instance:
        return _telemetry_instance
        
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    _telemetry_instance = TelemetryManager(service_name, endpoint)
    return _telemetry_instance

def get_telemetry() -> TelemetryManager:
    global _telemetry_instance
    if not _telemetry_instance:
        return init_telemetry()
    return _telemetry_instance
