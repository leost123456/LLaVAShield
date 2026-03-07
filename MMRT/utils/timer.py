from time import perf_counter

class Timer:
    def __init__(self, logger=None):
        self.times = []
        self.logger = logger

    def measure(self, label, fn, *args, **kwargs):
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        dt = perf_counter() - t0
        self.times.append(dt)
        if self.logger:
            self.logger.info("%s took %.2fs (avg: %.2fs over %d tasks)",
                             label, dt, self.avg, len(self.times))
        return out

    @property
    def avg(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0
