from dataclasses import dataclass

@dataclass(frozen=True, slots=True) 
class AttackerConfig:
    num_candidates: int = 2
    temperature: float = 1.5
    top_p: float = 0.95
    max_tokens: int = 4096

class AttackerModel:
    def __init__(self, config: AttackerConfig | None = None):
        self.config = config or AttackerConfig()
        self.max_retries = 2
        
    def generate(self, task_id, task, image_ids, captions, max_rounds, round_number, last_response, score):
        raise NotImplementedError()
