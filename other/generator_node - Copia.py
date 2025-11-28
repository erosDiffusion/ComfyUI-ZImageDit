
import torch
import numpy as np

# from comfy.model_management import register_custom_node
from PIL import Image

class ZImageTurboQDiTGenerateUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ZIMAGE_PIPELINE",),
                "prompt": ("STRING", {"default": "Young Chinese woman in red Hanfu"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "num_inference_steps": ("INT", {"default": 9, "min": 1, "max": 20}),
                "guidance_scale": ("FLOAT", {"default": 0.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image (Turbo)"

    def generate(self, pipe, prompt, height, width, num_inference_steps, guidance_scale, seed):
        if guidance_scale != 0.0:
            guidance_scale = 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(seed)

        out = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                   guidance_scale=guidance_scale, generator=generator)

        img: Image.Image = out.images[0]
        arr = np.array(img).astype(np.uint8)

        # Automatic unload
        try:
            del pipe.transformer
            if hasattr(pipe, "text_encoder"):
                del pipe.text_encoder
            if hasattr(pipe, "vae"):
                del pipe.vae
            torch.cuda.empty_cache()
            print("[Z-Image Turbo Q-DiT] Models unloaded and CUDA cache cleared.")
        except Exception as e:
            print(f"Unload failed: {e}")

        return (arr,)

# register_custom_node(ZImageTurboQDiTGenerateUnload)
