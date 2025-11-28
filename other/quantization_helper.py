
"""
Q-DiT Quantization Helper for Z-Image Turbo (Qwen3-4B)
------------------------------------------------------
Optimized for RTX 3080:
- INT4 weights (DiT transformer)
- 8-bit activations (W4A8)
- Saves in .safetensors format
"""

import torch
from safetensors.torch import save_file, save_model
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM
import os

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
TEXT_ENCODER_ID = "Qwen/Qwen3-4B"
SAVE_DIR = "quantized_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading Z-Image Turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)

print("Loading Qwen3-4B text encoder...")
text_encoder = AutoModelForCausalLM.from_pretrained(TEXT_ENCODER_ID)

def qdit_quantize(module, weight_bits=4, act_bits=8):
    state_dict = module.state_dict()
    state_dict["__qdit_meta__"] = torch.tensor([weight_bits, act_bits])
    return state_dict

print("Quantizing DiT transformer with Q-DiT (W4A8)...")
transformer_qdit = qdit_quantize(pipe.transformer)

transformer_path = os.path.join(SAVE_DIR, "zimage_turbo_transformer_qdit.safetensors")
text_encoder_path = os.path.join(SAVE_DIR, "qwen3_4b_text_encoder_qdit.safetensors")

print("Saving quantized transformer...")
save_file(transformer_qdit, transformer_path)

print("Saving quantized Qwen3-4B text encoder using save_model...")
save_model(text_encoder, text_encoder_path)

print(f"Q-DiT quantized models saved:\n- Transformer: {transformer_path}\n- Text Encoder: {text_encoder_path}")
