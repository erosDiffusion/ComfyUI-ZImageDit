Z-Image Turbo Q-DiT Bundle (Qwen3-4B)
=====================================

This custom node suite provides optimized methods to run Z-Image Turbo in ComfyUI, focusing on low memory usage via quantization.

## Features
1.  **Modular Lightweight Nodes**: Load quantized components (Transformer, CLIP, VAE) separately to minimize memory footprint.
    - **Format**: Uses the **Original Diffusers Format** (quantized).
    - **Sampler**: Requires `Z-Image Quanto Sampler`.
2.  **Comfy-Native Quantization**: Quantize any loaded ComfyUI model on the fly.
3.  **Quantization Script**: Generates compatible Diffusers-format quantized weights.

## Defaults & Paths
- **Quantized Models**: `models/quantized_models/`
- **DType**: `bfloat16` (Recommended for RTX 3080/Ampere)
- **Format**: `.safetensors` (Diffusers layout)

## Installation

1.  **Dependencies**:
    Ensure your ComfyUI environment has the required packages:
    `pip install diffusers transformers safetensors quanto`

2.  **Virtual Environment (for Quantization Script)**:
    The quantization script (`quantize_zimage.py`) should be run in a separate environment (e.g., the `.venv` in this folder) if your ComfyUI python doesn't have internet access or if you prefer isolation.
    - Activate venv: `venv\Scripts\activate`
    - Install deps: `pip install -r requirements.txt`

## Process 1: Lightweight Modular Nodes (Recommended)
This method uses the least amount of memory during loading and inference.

### Step 1: Generate Quantized Weights
Run the quantization script to download the model and save quantized `.safetensors` files.
**Note**: Run this using the `.venv` or a python environment with internet access.

```bash
# Inside the custom_nodes/ComfyUI-ZImageDit folder
..\..\..\python_embeded\python.exe quantize_zimage.py
# OR if using the local venv:
venv\Scripts\python.exe quantize_zimage.py
```
This will create `models/quantized_models/` containing:
- `zimage_turbo_transformer_qdit.safetensors`
- `qwen_text_encoder_qdit.safetensors`

### Step 2: Usage in ComfyUI
1.  **Load Z-Image Transformer (Quanto)**:
    - Point `transformer_path` to `models/quantized_models/zimage_turbo_transformer_qdit.safetensors`.
2.  **Load Z-Image CLIP (Quanto)**:
    - Point `text_encoder_path` to `models/quantized_models/qwen_text_encoder_qdit.safetensors`.
3.  **Load Z-Image VAE**:
    - Loads the VAE (standard float32/bf16).
4.  **Z-Image Quanto Sampler**:
    - Connect the `transformer`, `text_encoder`, and `vae` outputs to this node.
    - Set your prompt and parameters.
    - Connect output to `Save Image` or `Preview Image`.
    - **IMPORTANT**: Do NOT use the standard ComfyUI `KSampler` with these nodes. They return `diffusers` objects which are incompatible with KSampler. You MUST use `Z-Image Quanto Sampler`.

## Process 2: Comfy-Native Quantization
Use this if you want to quantize a model that is already loaded in ComfyUI (e.g., via standard loaders).
**Note**: This requires enough RAM to load the full model first.

1.  Load your model using standard ComfyUI nodes (e.g., `Load Checkpoint` or `Load Z-Image Turbo (Q-DiT)`).
2.  Add **Quantize Active Model (Quanto)** node.
3.  Connect the `MODEL` output to the quantizer.
4.  Select `weights` (e.g., `int8`, `int4`) and `activations` (e.g., `int8`).
5.  Connect the output `MODEL` to your sampler (e.g., `KSampler` or `Z-Image Generate`).

### Advanced: Hybrid Quantization
Use **Quantize Model Hybrid (Quanto)** for fine-grained control.
- **Linear Weights**: Set quantization for linear layers (e.g., `int4`).
- **Conv Weights**: Set quantization for conv layers (e.g., `int8`).
- **Keep First/Last**: Keeps input/output layers in full precision (recommended).
- **Exclude Patterns**: Comma-separated list of layer names to exclude (e.g., `norm, bias`).

## Node Descriptions

- **Load Z-Image Transformer (Quanto)**: Loads the Z-Image DiT structure and fills it with quantized weights.
- **Load Z-Image CLIP (Quanto)**: Loads the Qwen3 text encoder structure and fills it with quantized weights.
- **Load Z-Image VAE**: Loads the VAE component.
- **Z-Image Quanto Sampler**: A custom sampler designed to work with the `diffusers` objects returned by the modular loaders.
- **Quantize Active Model (Quanto)**: Applies `quanto` quantization to a standard ComfyUI `MODEL` object.

## Hardware Notes (RTX 3080)
- **DType**: Use `bfloat16` (default) for best stability and performance.
- **Quantization**: `int4` weights and `int8` activations (W4A8) provide a good balance of quality and speed on Ampere GPUs.
