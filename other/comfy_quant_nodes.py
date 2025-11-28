import torch
import quanto
import comfy.model_management

class QuantizeActiveModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "weights": (["int8", "int4", "float8"], {"default": "int8"}),
                "activations": (["none", "int8", "float8"], {"default": "int8"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    CATEGORY = "Z-Image (Quantization)"

    def quantize(self, model, weights, activations):
        print(f"Quantizing Active Model (Weights: {weights}, Activations: {activations})...")
        
        # Clone the model wrapper to avoid affecting other nodes using the same model instance?
        # ComfyUI models are usually shared. If we quantize in place, it affects everything.
        # But `model.clone()` might be shallow.
        # For safety, we should probably clone the underlying diffusion model if we want isolation,
        # but that doubles memory usage before quantization.
        # Given the goal is memory reduction, we might want to quantize in place, but warn the user.
        # However, usually users want to load -> quantize -> use.
        
        # Let's try to clone the `model` object (wrapper) and the `diffusion_model` inside it.
        new_model = model.clone()
        
        # Access the underlying torch module
        # In ComfyUI, `model.model.diffusion_model` is usually the UNet/DiT
        diffusion_model = new_model.model.diffusion_model
        
        # Map string args to quanto types
        w_map = {"int8": quanto.qint8, "int4": quanto.qint4, "float8": quanto.qfloat8}
        a_map = {"none": None, "int8": quanto.qint8, "float8": quanto.qfloat8}
        
        w_type = w_map.get(weights)
        a_type = a_map.get(activations)
        
        # Quantize
        # We need to ensure the model is on a device that quanto supports (CPU/CUDA).
        # It's likely on CPU or GPU depending on Comfy's management.
        
        quanto.quantize(diffusion_model, weights=w_type, activations=a_type)
        quanto.freeze(diffusion_model)
        
        print("Model quantized and frozen.")
        
        return (new_model,)

class LoadZImageComfyQuantized:
    """
    Experimental node to load Z-Image and quantize immediately.
    This is a placeholder for a more advanced implementation if needed.
    For now, users can use standard loaders + QuantizeActiveModel.
    """
class QuantizeComfyModelHybrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "linear_weights": (["int8", "int4", "float8", "none"], {"default": "int4"}),
                "conv_weights": (["int8", "int4", "float8", "none"], {"default": "int8"}),
                "activations": (["none", "int8", "float8"], {"default": "int8"}),
                "keep_first_last": ("BOOLEAN", {"default": True}),
                "exclude_patterns": ("STRING", {"default": "norm, bias, time_emb"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize_hybrid"
    CATEGORY = "Z-Image (Quantization)"

    def quantize_hybrid(self, model, linear_weights, conv_weights, activations, keep_first_last, exclude_patterns):
        print(f"Hybrid Quantization: Linear={linear_weights}, Conv={conv_weights}, Act={activations}")
        
        new_model = model.clone()
        diffusion_model = new_model.model.diffusion_model
        
        # Helper to map string to quanto type
        def get_qtype(name):
            if name == "none": return None
            if name == "int8": return quanto.qint8
            if name == "int4": return quanto.qint4
            if name == "float8": return quanto.qfloat8
            return None

        w_linear = get_qtype(linear_weights)
        w_conv = get_qtype(conv_weights)
        a_type = get_qtype(activations)
        
        excludes = [p.strip() for p in exclude_patterns.split(",") if p.strip()]
        
        # Identify layers to quantize
        # We traverse the model and apply quantization selectively
        
        # Get all named modules first to identify first/last if needed
        # But "first" and "last" are hard to define generically in a graph.
        # Usually "first" is the input projection (conv_in) and "last" is output (conv_out).
        # We can look for specific names often used in DiT/UNet.
        
        # Common names for input/output in diffusers/comfy:
        # x_embedder, final_layer, conv_in, conv_out
        
        first_last_names = ["x_embedder", "final_layer", "conv_in", "conv_out", "pos_embed"]
        
        for name, module in diffusion_model.named_modules():
            # Skip if module is not a leaf or not a quantizable layer type
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
                
            # Check excludes
            if any(ex in name for ex in excludes):
                continue
                
            # Check first/last
            if keep_first_last:
                if any(fl in name for fl in first_last_names):
                    continue
            
            # Determine weights type
            w_target = None
            if isinstance(module, torch.nn.Linear):
                w_target = w_linear
            elif isinstance(module, torch.nn.Conv2d):
                w_target = w_conv
            
            if w_target is None:
                continue
                
            # Apply quantization to this specific module
            # quanto.quantize modifies in-place, but we need to target just this module.
            # We can pass the module directly.
            quanto.quantize(module, weights=w_target, activations=a_type)
            
        quanto.freeze(diffusion_model)
        print("Hybrid quantization complete.")
        
        return (new_model,)
