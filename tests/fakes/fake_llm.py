"""FakeLLM — deterministic LLM service for tests."""


class FakeLLM:
    def __init__(self, response="ok"):
        self.response = response
        self.calls = 0

    async def complete(self, *a, **k):
        self.calls += 1
        return self.response, {"model": "fake", "total_cost": 0.0}

    async def generate_structured(self, *a, **k):
        self.calls += 1
        return self.response, {"model": "fake", "total_cost": 0.0}
