import os
import ray
from typing import List

@ray.remote
class SD3Worker:
    def __init__(self):
        try:
            gpu_ids = ray.get_gpu_ids() 
        except Exception:
            gpu_ids = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids)) 

class RaySD3Pool:
    def __init__(self, gpus: int, slots_per_gpu: int = 1, actors: int | None = None):
        assert gpus >= 1 and slots_per_gpu >= 1
        capacity = gpus * slots_per_gpu
        total_actors = capacity if actors is None else min(actors, capacity)
        fraction = 1.0 / float(slots_per_gpu)

        self.actors: List[ray.actor.ActorHandle] = [
            SD3Worker.options(num_gpus=fraction, max_concurrency=1).remote()
            for _ in range(total_actors)
        ]
        self._rr = 0

    def submit(self, prompt: str, save_path: str):
        actor = self.actors[self._rr]
        self._rr = (self._rr + 1) % len(self.actors)
        return actor.generate.remote(prompt, save_path) 

    def get(self, obj_ref, timeout: float | None = 600.0):
        return ray.get(obj_ref, timeout=timeout)

    def map_generate(self, jobs: List[tuple[str, str]]):
        return [self.submit(prompt, save_path) for prompt, save_path in jobs]
