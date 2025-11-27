import torch
import os
import quanto
from diffusers import DiffusionPipeline, Transformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

def quantize_and_save():
    MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
    SAVE_DIR = "models/quantized_models"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Starting quantization for {MODEL_ID}...")
    
    # 1. Quantize Text Encoder (Qwen3-4B)
    print("Loading Text Encoder...")
    try:
        text_encoder = AutoModelForCausalLM.from_pretrained(MODEL_ID, subfolder="text_encoder", trust_remote_code=True, dtype=torch.bfloat16)
    except Exception:
        print("Could not load from subfolder, trying direct Qwen/Qwen3-4B...")
        text_encoder = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True, dtype=torch.bfloat16)

    print("Quantizing Text Encoder (Weights: int4, Activations: int8)...")
    # Collect modules first to avoid modifying graph while iterating (prevents RecursionError)
    modules_to_quantize = []
    for name, module in text_encoder.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            modules_to_quantize.append(module)
            
    for module in modules_to_quantize:
        quanto.quantize(module, weights=quanto.qint4, activations=quanto.qint8)
    quanto.freeze(text_encoder)
    
    te_path = os.path.join(SAVE_DIR, "qwen_text_encoder_qdit.safetensors")
    print(f"Saving Text Encoder to {te_path}...")
    # Save state dict, filtering out non-tensor values (quanto metadata)
    te_state_dict = {k: v for k, v in text_encoder.state_dict().items() if isinstance(v, torch.Tensor)}
    save_file(te_state_dict, te_path)
    del text_encoder
    del te_state_dict
    
    # 2. Quantize Transformer
    print("Loading Transformer...")
    # Try to load ZImageTransformer2DModel
    try:
        from diffusers import ZImageTransformer2DModel
        transformer = ZImageTransformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    except ImportError:
        print("ZImageTransformer2DModel not found. Using Transformer2DModel (might fail if architecture is custom)...")
        transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
        
    print("Quantizing Transformer (Weights: int4, Activations: int8)...")
    # Selective quantization to avoid LayerNorm crash and improve quality
    modules_to_quantize = []
    for name, module in transformer.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            modules_to_quantize.append(module)
            
    for module in modules_to_quantize:
        quanto.quantize(module, weights=quanto.qint4, activations=quanto.qint8)
    quanto.freeze(transformer)
    
    tr_path = os.path.join(SAVE_DIR, "zimage_turbo_transformer_qdit.safetensors")
    print(f"Saving Transformer to {tr_path}...")
    tr_state_dict = {k: v for k, v in transformer.state_dict().items() if isinstance(v, torch.Tensor)}
    save_file(tr_state_dict, tr_path)
    del transformer
    del tr_state_dict
    
    print("Quantization complete!")

if __name__ == "__main__":
    quantize_and_save()
