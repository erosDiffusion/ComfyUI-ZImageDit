import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from diffusers.schedulers import DDIMScheduler

class ZImageTurboQDiTGenerateUnload:
    """
    Offline generator that takes (MODEL, VAE, TEXT_ENCODER) and runs a DDIM sampling loop.
    Assumes transformer forward signature similar to UNet2DConditionModel; if you're using
    QwenImageTransformer2DModel, this may still work depending on the forward signature.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "text_encoder": ("TEXT_ENCODER",),
                "prompt": ("STRING", {"default": "a beautiful landscape"}),
                "negative_prompt": ("STRING", {"default": ""}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image (Offline)"

    def generate(self, model, vae, text_encoder, prompt, negative_prompt, height, width, steps, guidance_scale, seed):
        device = next(model.parameters()).device
        torch.manual_seed(seed)

        # Tokenizer (Qwen tokenizer; if not available, use a generic AutoTokenizer from Qwen repo)
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")  # local cache if available
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B", local_files_only=False)

        # Encode prompts
        pos_inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
        neg_inputs = tokenizer([negative_prompt], return_tensors="pt", padding=True).to(device)

        pos_embeds = text_encoder(**pos_inputs).last_hidden_state
        neg_embeds = text_encoder(**neg_inputs).last_hidden_state

        # Scheduler
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        scheduler.set_timesteps(steps, device=device)

        # Latent size from image size (standard 8x compression)
        latent_h = height // 8
        latent_w = width // 8
        latents = torch.randn((1, getattr(model, "in_channels", 4), latent_h, latent_w), device=device)

        # DDIM loop with classifier-free guidance
        for t in scheduler.timesteps:
            latent_input = latents
            # unconditional
            noise_uncond = model(latent_input, t, encoder_hidden_states=neg_embeds).sample
            # conditional
            noise_text = model(latent_input, t, encoder_hidden_states=pos_embeds).sample
            # CFG
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode with VAE
        # SD convention: scale by 1/0.18215 if VAE expects normalized latents.
        try:
            image = vae.decode(latents / 0.18215).sample
        except Exception:
            image = vae.decode(latents).sample

        image = (image.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        image = (image * 255).byte().cpu().numpy()
        pil_img = Image.fromarray(image[0].transpose(1, 2, 0))
        return (np.array(pil_img),)

