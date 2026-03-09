import os
import gc
import torch
from diffusers import StableDiffusion3Pipeline

from ....config import MODEL_PATHS

_SD3_PIPE = None  

def _pick_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:7")
    return torch.device("cpu")

def _load_sd3(device: str | None = None,
              dtype: torch.dtype = torch.float16,
              model_path: str = None):
    global _SD3_PIPE
    if _SD3_PIPE is not None:
        return _SD3_PIPE

    # SD3-Medium model path
    if model_path:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    dev = _pick_device(device)
    pipe = pipe.to(dev)
    _SD3_PIPE = pipe
    return _SD3_PIPE

def _unload_sd3():
    global _SD3_PIPE
    if _SD3_PIPE is None:
        return
    del _SD3_PIPE
    _SD3_PIPE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def SD3_medium_generate(prompt: str,
                        save_path: str,
                        model_path: str = MODEL_PATHS["stable-diffusion-3.5-medium"],
                        device: str | None = None,
                        steps: int = 30,
                        guidance: float = 4.5,
                        seed: int | None = None,
                        unload_after: bool = False) -> str:
    pipe = _load_sd3(model_path=model_path, device=device)
    generator = None
    try:
        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
    except Exception:
        generator = None

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )
    image = out.images[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

    if unload_after:
        _unload_sd3()

    return save_path