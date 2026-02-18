"""
KernelTelemetry for SGR Kernel.
Tracks architectural metrics (not just LLM tokens).
"""
import time
from typing import Dict, Any

class KernelTelemetry:
    """
    Tracks kernel performance and reliability metrics.
    """
    def __init__(self):
        self.metrics = {
            "plan_latency": [],
            "step_latency": [],
            "retry_count": 0,
            "repair_count": 0,
            "escalation_count": 0,
            "skill_success_count": 0,
            "checkpoint_count": 0,
            "resume_count": 0
        }
    
    def record_latency(self, metric: str, duration: float):
        if metric in self.metrics and isinstance(self.metrics[metric], list):
             self.metrics[metric].append(duration)

    def increment(self, metric: str):
        if metric in self.metrics and isinstance(self.metrics[metric], int):
            self.metrics[metric] += 1

    def get_snapshot(self) -> Dict[str, Any]:
        return self.metrics.copy()
