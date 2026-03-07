from dataclasses import dataclass

@dataclass(frozen=True, slots=True) 
class EvaluatorConfig:
    temperature: float = 0.0
    max_tokens: int = 512

class EvaluatorModel:
    def __init__(self, config: EvaluatorConfig | None = None):
        self.config = config or EvaluatorConfig()
        self.max_retries = 2

    def evaluate(self, task, target_history):
        raise NotImplementedError()