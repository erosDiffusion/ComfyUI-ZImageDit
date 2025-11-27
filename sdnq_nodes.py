import torch
import numpy as np
from PIL import Image
import subprocess
import sys
from diffusers import DiffusionPipeline

# Try to import sdnq, install with workaround if not present
try:
    import sdnq
except ImportError:
    print("SDNQ not found. Attempting installation with workaround...")
    try:
        # Try installing from PyPI first (if available)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sdnq", "--no-cache-dir"])
    except:
        print("PyPI install failed. Trying git installation with --no-build-isolation...")
        try:
            # Work around broken pyproject.toml by skipping build isolation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/Disty0/sdnq",
                "--no-build-isolation"
            ])
        except Exception as e:
            print(f"SDNQ installation failed: {e}")
            print("MANUAL INSTALLATION REQUIRED:")
            print("1. Clone: git clone https://github.com/Disty0/sdnq")
            print("2. Fix pyproject.toml: change 'license = \"GPL-3.0-only\"' to 'license = {text = \"GPL-3.0-only\"}'")
            print("3. Install: pip install ./sdnq")
            raise ImportError("SDNQ installation failed. Please install manually (see instructions above).")
    
    # Try importing again
    try:
        import sdnq
    except ImportError:
        raise ImportError("SDNQ installation failed. Please install manually.")

class LoadZImageSDNQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
                    "tooltip": "Hugging Face model ID for the SDNQ-quantized Z-Image Turbo model"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load the model on. 'auto' selects CUDA if available, otherwise CPU"
                }),
            },
            "optional": {
                "attention_backend": (["default", "flash", "flash3", "sage"], {
                    "default": "default",
                    "tooltip": "Attention mechanism: 'default' (SDPA), 'flash' (Flash Attention 2), 'flash3' (Flash Attention 3), 'sage' (Sage Attention). Flash/Sage may improve speed on supported GPUs"
                }),
                "enable_compilation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compile transformer with torch.compile for faster inference. First run will be slower while compiling"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable sequential CPU offloading to reduce VRAM usage. Models are moved between CPU and GPU as needed"
                }),
                "low_cpu_mem_usage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use memory-efficient loading method. Recommended to keep enabled"
                }),
                "vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Process VAE decode in tiles to reduce VRAM usage. Useful for high resolutions"
                }),
            }
        }

    RETURN_TYPES = ("ZIMAGE_SDNQ_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Z-Image (SDNQ)"

    def load_pipeline(self, model_id, device, attention_backend="default", enable_compilation=False, cpu_offload=False, low_cpu_mem_usage=True, vae_tiling=False):
        print(f"Loading SDNQ Pipeline from {model_id}...")
        
        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
            
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        
        # Apply attention backend
        if attention_backend != "default":
            print(f"Setting attention backend to: {attention_backend}")
            try:
                if attention_backend == "flash":
                    pipeline.transformer.set_attention_backend("flash")
                elif attention_backend == "flash3":
                    pipeline.transformer.set_attention_backend("_flash_3")
                elif attention_backend == "sage":
                    pipeline.transformer.set_attention_backend("sage")
            except Exception as e:
                print(f"Warning: Failed to set attention backend '{attention_backend}': {e}")
        
        # Apply CPU offload if requested
        if cpu_offload:
            print("Enabling CPU offload...")
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(dev)
        
        # Enable VAE tiling if requested (reduces VRAM usage during decode)
        if vae_tiling:
            print("Enabling VAE tiling...")
            try:
                pipeline.enable_vae_tiling()
            except Exception as e:
                print(f"Warning: VAE tiling failed: {e}")
        
        # Compile if requested (must be done after moving to device)
        if enable_compilation:
            print("Compiling transformer (first run will be slower)...")
            try:
                pipeline.transformer.compile()
            except Exception as e:
                print(f"Warning: Compilation failed: {e}")
        
        print("SDNQ Pipeline loaded successfully.")
        return (pipeline,)

# Resolution presets from official Gradio app
RES_CHOICES = {
    "1024": [
        "1024x1024 ( 1:1 )",
        "1152x896 ( 9:7 )",
        "896x1152 ( 7:9 )",
        "1152x864 ( 4:3 )",
        "864x1152 ( 3:4 )",
        "1248x832 ( 3:2 )",
        "832x1248 ( 2:3 )",
        "1280x720 ( 16:9 )",
        "720x1280 ( 9:16 )",
        "1344x576 ( 21:9 )",
        "576x1344 ( 9:21 )",
    ],
    "1280": [
        "1280x1280 ( 1:1 )",
        "1440x1120 ( 9:7 )",
        "1120x1440 ( 7:9 )",
        "1472x1104 ( 4:3 )",
        "1104x1472 ( 3:4 )",
        "1536x1024 ( 3:2 )",
        "1024x1536 ( 2:3 )",
        "1600x896 ( 16:9 )",
        "896x1600 ( 9:16 )",
        "1680x720 ( 21:9 )",
        "720x1680 ( 9:21 )",
    ],
}

# Flatten all presets for dropdown
ALL_RESOLUTIONS = ["custom"]
for cat_resolutions in RES_CHOICES.values():
    ALL_RESOLUTIONS.extend(cat_resolutions)

class ZImageSDNQGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ZIMAGE_SDNQ_PIPELINE",),
                "prompt": ("STRING", {
                    "default": "a beautiful landscape",
                    "multiline": True,
                    "tooltip": "Text description of the image to generate. Be descriptive for best results"
                }),
                "resolution_preset": (ALL_RESOLUTIONS, {
                    "default": "1024x1024 ( 1:1 )",
                    "tooltip": "Select a resolution preset or choose 'custom' to set manual width/height"
                }),
                "num_inference_steps": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps. Z-Image Turbo is optimized for 5-9 steps. More steps = slower but potentially higher quality"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Z-Image Turbo is trained for guidance_scale=0.0 (recommended). Higher values increase prompt adherence but may reduce quality"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation. Same seed + settings = same image"
                }),
                "shift": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Time shift for FlowMatch scheduler. Controls sampling distribution: lower (1-2) = more noise-end sampling, higher (4-10) = more clean-end sampling. Default 3.0 is balanced"
                }),
                "max_sequence_length": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 2048,
                    "step": 128,
                    "tooltip": "Maximum prompt token length. 1024 recommended for detailed prompts (Z-Image works best with long descriptions). 512 for faster speed"
                }),
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Custom output width (only used when resolution_preset is 'custom'). Must be divisible by 64"
                }),
                "custom_height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Custom output height (only used when resolution_preset is 'custom'). Must be divisible by 64"
                }),
                "enhance_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Z-Image's built-in prompt enhancer for improved semantic understanding and detail. May not be available in all models"
                }),
                "unload_after_generation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move models to CPU after generation to free VRAM. Useful for memory-constrained setups"
                }),
                "gc_cuda": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run Python garbage collection and clear CUDA cache after generation. Helps reclaim memory"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image (SDNQ)"

    def generate(self, pipeline, prompt, resolution_preset, num_inference_steps, guidance_scale, seed, shift, max_sequence_length, custom_width=1024, custom_height=1024, enhance_prompt=False, unload_after_generation=False, gc_cuda=False):
        import gc
        
        # Parse resolution from preset or use custom values
        if resolution_preset == "custom":
            width = custom_width
            height = custom_height
            print(f"Using custom resolution: {width}x{height}")
        else:
            # Parse preset string format: "1024x1024 ( 1:1 )"
            try:
                resolution_str = resolution_preset.split(" ")[0]  # Get "1024x1024"
                width, height = map(int, resolution_str.split("x"))
                print(f"Using preset resolution: {width}x{height} from '{resolution_preset}'")
            except Exception as e:
                print(f"Failed to parse resolution preset '{resolution_preset}': {e}. Using 1024x1024")
                width, height = 1024, 1024
        
        print(f"Generating image with SDNQ pipeline (shift={shift}, max_seq_len={max_sequence_length})...")
        
        # Note: Prompt enhancer is not available in SDNQ models (it's a separate API service)
        # Skip the enhancement attempt to avoid console spam
        
        generator = torch.manual_seed(seed)
        
        # Configure shift on the scheduler before generation
        original_shift = None
        if hasattr(pipeline.scheduler, 'config'):
            try:
                original_shift = pipeline.scheduler.config.shift
                pipeline.scheduler.config.shift = shift
                print(f"Configured scheduler shift: {shift}")
            except Exception as e:
                print(f"Warning: Could not configure shift on scheduler: {e}")
        
        # Z-Image uses FlowMatchEulerDiscreteScheduler
        # shift must be configured on the scheduler, not passed to __call__
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]
        
        # Restore original shift value if we changed it
        if original_shift is not None and hasattr(pipeline.scheduler, 'config'):
            try:
                pipeline.scheduler.config.shift = original_shift
            except:
                pass
        
        # Unload models if requested
        if unload_after_generation:
            print("Unloading models to free VRAM...")
            try:
                # Move models to CPU
                if hasattr(pipeline, 'transformer'):
                    pipeline.transformer.to('cpu')
                if hasattr(pipeline, 'text_encoder'):
                    pipeline.text_encoder.to('cpu')
                if hasattr(pipeline, 'vae'):
                    pipeline.vae.to('cpu')
            except Exception as e:
                print(f"Model unloading failed: {e}")
        
        # CUDA garbage collection if requested
        if gc_cuda:
            print("Running CUDA garbage collection...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Convert PIL to tensor
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)
