
import torch
# from comfy.model_management import register_custom_node
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
import os

class LoadZImageTurboQDiTOffline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "transformer_path": ("STRING", {"default": "quantized_models/zimage_turbo_transformer_qdit.safetensors"}),
                "text_encoder_path": ("STRING", {"default": "quantized_models/qwen_text_encoder_qdit.safetensors"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "Z-Image (Turbo)"

    def load(self, model_id, transformer_path, text_encoder_path, dtype, device):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True)

        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
        pipe.to(dev)

        print("Loading Q-DiT quantized transformer from .safetensors...")
        state_dict_transformer = load_file(transformer_path)
        pipe.transformer.load_state_dict({k: v for k, v in state_dict_transformer.items() if not k.startswith("__qdit_meta__")})

        if hasattr(pipe, "text_encoder") and os.path.exists(text_encoder_path):
            print("Loading Q-DiT quantized text encoder from .safetensors...")
            state_dict_text = load_file(text_encoder_path)
            pipe.text_encoder.load_state_dict({k: v for k, v in state_dict_text.items() if not k.startswith("__qdit_meta__")})

        return (pipe,)

# register_custom_node(LoadZImageTurboQDiTOffline)
