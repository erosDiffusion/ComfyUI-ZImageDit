import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler, Transformer2DModel, AutoencoderKL
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file
from PIL import Image
import numpy as np
import quanto

# --- Modular Loaders ---

class LoadQuantoZImageTransformer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "transformer_path": ("STRING", {"default": "models/quantized_models/zimage_turbo_transformer_qdit.safetensors"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_TRANSFORMER",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "load_transformer"
    CATEGORY = "Z-Image (Modular)"

    def load_transformer(self, model_id, transformer_path, dtype, device):
        print(f"Loading Quantized Transformer from {transformer_path}...")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        # Try to import ZImageTransformer2DModel
        try:
            from diffusers import ZImageTransformer2DModel
            TransformerClass = ZImageTransformer2DModel
        except ImportError:
            print("ZImageTransformer2DModel not found. Using Transformer2DModel.")
            TransformerClass = Transformer2DModel

        try:
            config = TransformerClass.load_config(model_id, subfolder="transformer")
            with torch.device("meta"):
                transformer = TransformerClass.from_config(config)
        except Exception as e:
            print(f"Failed to load transformer config: {e}")
            raise e
        
        transformer.to_empty(device="cpu")
        
        print("Initializing Quantized Structure (quanto)...")
        modules_to_quantize = []
        for name, module in transformer.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                modules_to_quantize.append(module)
        
        for module in modules_to_quantize:
            quanto.quantize(module, weights=quanto.qint4, activations=quanto.qint8)
        
        print("Loading Weights (Manual Assignment)...")
        state_dict = load_file(transformer_path)
        
        # Manually assign weights to bypass strict checks in quanto's load_state_dict hook
        # This is necessary because we filtered out metadata keys during saving
        model_dict = transformer.state_dict()
        for name, param in transformer.named_parameters():
            if name in state_dict:
                # We assume the shapes match. 
                # If param is a QTensor, we might need to handle it carefully, 
                # but usually assigning to .data works if the tensor is compatible.
                try:
                    param.data = state_dict[name].to(param.device)
                except Exception as e:
                    print(f"Failed to assign {name}: {e}")
            elif name.endswith("_scale") and name in state_dict:
                 # Handle scales if they are separate parameters/buffers
                 pass
        
        # Also handle buffers (like scales if they are buffers)
        for name, buf in transformer.named_buffers():
            if name in state_dict:
                try:
                    buf.data = state_dict[name].to(buf.device)
                except Exception as e:
                    print(f"Failed to assign buffer {name}: {e}")

        del state_dict
        
        quanto.freeze(transformer)
        transformer.to(dev)
        
        return (transformer,)

class LoadQuantoZImageCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "text_encoder_path": ("STRING", {"default": "models/quantized_models/qwen_text_encoder_qdit.safetensors"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_CLIP",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_clip"
    CATEGORY = "Z-Image (Modular)"

    def load_clip(self, model_id, text_encoder_path, dtype, device):
        print(f"Loading Quantized Text Encoder from {text_encoder_path}...")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        print("Loading Config (Qwen3)...")
        try:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
        except Exception:
             print("Warning: Could not load Qwen/Qwen3-4B config directly. Trying from model_id...")
             config = AutoConfig.from_pretrained(model_id, subfolder="text_encoder")
        
        with torch.device("meta"):
             text_encoder = AutoModelForCausalLM.from_config(config)
        
        text_encoder.to_empty(device="cpu")

        print("Initializing Quantized Structure (quanto)...")
        modules_to_quantize = []
        for name, module in text_encoder.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                modules_to_quantize.append(module)
        
        for module in modules_to_quantize:
            quanto.quantize(module, weights=quanto.qint4, activations=quanto.qint8)
        
        print("Loading Weights (Manual Assignment)...")
        state_dict = load_file(text_encoder_path)
        
        # Manually assign weights to bypass strict checks
        for name, param in text_encoder.named_parameters():
            if name in state_dict:
                try:
                    param.data = state_dict[name].to(param.device)
                except Exception as e:
                    print(f"Failed to assign {name}: {e}")
        
        for name, buf in text_encoder.named_buffers():
            if name in state_dict:
                try:
                    buf.data = state_dict[name].to(buf.device)
                except Exception as e:
                    print(f"Failed to assign buffer {name}: {e}")
        
        del state_dict
        
        quanto.freeze(text_encoder)
        text_encoder.to(dev, dtype=torch_dtype)
        
        return (text_encoder,)

class LoadQuantoZImageVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Z-Image (Modular)"

    def load_vae(self, model_id, dtype, device):
        print(f"Loading VAE from {model_id}...")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else (torch.float16 if dtype == "float16" else torch.float32)
        
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
            
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)
        vae.to(dev)
        
        return (vae,)

class ZImageQuantoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("ZIMAGE_TRANSFORMER",),
                "text_encoder": ("ZIMAGE_CLIP",),
                "vae": ("ZIMAGE_VAE",),
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
    CATEGORY = "Z-Image (Modular)"

    def generate(self, transformer, text_encoder, vae, prompt, negative_prompt, height, width, steps, guidance_scale, seed):
        torch.manual_seed(seed)
        device = transformer.device
        
        # Assemble a temporary pipeline for generation
        # We need the tokenizer too. Ideally this should be passed or loaded here.
        # For simplicity, we load it here (lightweight).
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B", local_files_only=False)
            
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        
        # We construct the pipeline
        # Note: We can't easily pass this to standard KSampler because it's a diffusers pipeline.
        # So we run the loop here.
        
        pipe = DiffusionPipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo", # Dummy ID to get structure if needed, or just use components
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            scheduler=scheduler,
            torch_dtype=transformer.dtype,
            device_map=None
        )
        pipe.to(device)
        
        print("Generating image...")
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                output_type="pil"
            ).images[0]
        except Exception as e:
            print(f"Pipeline call failed: {e}.")
            raise e

        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)
