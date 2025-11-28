
"""
Q-DiT Calibration Script for Z-Image Turbo (Qwen3-4B)
=====================================================
Performs activation-aware calibration for W4A8 quantization.
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

activation_stats = {}

def collect_stats(module, name):
    def hook(_, __, output):
        if name not in activation_stats:
            activation_stats[name] = []
        activation_stats[name].append(output.detach().cpu())
    return hook

for name, layer in pipe.transformer.named_modules():
    if hasattr(layer, 'forward'):
        layer.register_forward_hook(collect_stats(layer, name))

print("Running calibration passes...")
calib_prompts = ["A beautiful landscape", "Portrait of a woman", "Futuristic city"]
for prompt in calib_prompts:
    _ = pipe(prompt=prompt, height=512, width=512, num_inference_steps=2, guidance_scale=0.0)

activation_scales = {}
for name, tensors in activation_stats.items():
    concat = torch.cat(tensors, dim=0)
    max_val = concat.abs().max()
    scale = max_val / (2**7 - 1)
    activation_scales[name] = scale.item()

transformer_state = pipe.transformer.state_dict()
for k, v in transformer_state.items():
    if v.dtype in [torch.float32, torch.bfloat16]:
        v_q = torch.clamp(torch.round(v / (v.abs().max() / 7)), -8, 7).to(torch.int8)
        transformer_state[k] = v_q
transformer_state['__activation_scales__'] = torch.tensor(list(activation_scales.values()))

transformer_path = os.path.join(SAVE_DIR, "zimage_turbo_transformer_qdit_calibrated.safetensors")
text_encoder_path = os.path.join(SAVE_DIR, "qwen3_4b_text_encoder_qdit_calibrated.safetensors")

print("Saving calibrated transformer...")
save_file(transformer_state, transformer_path)

print("Saving calibrated Qwen3-4B text encoder using save_model...")
save_model(text_encoder, text_encoder_path)

print(f"Calibration complete:\n- Transformer: {transformer_path}\n- Text Encoder: {text_encoder_path}")
