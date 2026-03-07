import time
from dataclasses import dataclass

@dataclass
class TimeLimit:
    seconds: float
    start: float = None

    def __post_init__(self):
        if self.start is None:
            self.start = time.monotonic()

    def expired(self) -> bool:
        return (time.monotonic() - self.start) >= self.seconds

    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.start))

    def expires_at(self) -> float:
        return self.start + self.seconds