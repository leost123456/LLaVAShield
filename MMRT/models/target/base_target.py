from dataclasses import dataclass

@dataclass(frozen=True, slots=True) 
class TargetConfig:
    temperature: float = 0.0
    top_p: float = 0.0
    max_tokens: int = 4096

class TargetModel:
     def __init__(self, config: TargetConfig | None = None):
        self.config = config or TargetConfig()
        self.max_retries = 2

     def respond(self, task_id, next_question, image_ids, generated_image_ids,history):
        raise NotImplementedError()