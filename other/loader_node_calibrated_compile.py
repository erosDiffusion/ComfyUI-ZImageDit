
import torch

from diffusers import DiffusionPipeline
from safetensors.torch import load_file
import os

class LoadZImageTurboQDiTCalibratedCompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "transformer_path": ("STRING", {"default": "quantized_models/zimage_turbo_transformer_qdit_calibrated.safetensors"}),
                "text_encoder_path": ("STRING", {"default": "quantized_models/qwen_text_encoder_qdit_calibrated.safetensors"}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "enable_compile": ("BOOL", {"default": False}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "Z-Image (Turbo)"

    def load(self, model_id, transformer_path, text_encoder_path, dtype, device, enable_compile):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True)

        dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
        pipe.to(dev)

        print("Loading calibrated Q-DiT transformer from .safetensors...")
        state_dict_transformer = load_file(transformer_path)
        activation_scales = None
        if "__activation_scales__" in state_dict_transformer:
            activation_scales = state_dict_transformer.pop("__activation_scales__")
        pipe.transformer.load_state_dict(state_dict_transformer)

        if hasattr(pipe, "text_encoder") and os.path.exists(text_encoder_path):
            print("Loading calibrated Q-DiT text encoder from .safetensors...")
            state_dict_text = load_file(text_encoder_path)
            if "__activation_scales__" in state_dict_text:
                state_dict_text.pop("__activation_scales__")
            pipe.text_encoder.load_state_dict(state_dict_text)

        # Wrap transformer forward to apply activation scaling
        if activation_scales is not None:
            scales_list = activation_scales.tolist()
            original_forward = pipe.transformer.forward

            def scaled_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                if isinstance(output, torch.Tensor):
                    scale_factor = max(scales_list) if scales_list else 1.0
                    output = output / scale_factor
                return output

            pipe.transformer.forward = scaled_forward
            print("Activation scaling applied during inference.")

        # Optional torch.compile for DiT transformer
        if enable_compile:
            try:
                print("Compiling transformer with torch.compile (mode=reduce-overhead)...")
                pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=False)
                print("Compilation successful.")
            except Exception as e:
                print(f"torch.compile failed: {e}. Continuing without compilation.")

        return (pipe,)


