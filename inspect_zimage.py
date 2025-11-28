import inspect
import torch
import sys

try:
    # Try to import the class directly if possible
    # Based on PR link, it might be in diffusers.models.transformers.z_image_transformer_2d
    # But let's try to find it in diffusers.models
    from diffusers import DiffusionPipeline
    
    # We can try to load the pipeline (might be slow) or just inspect the module structure
    # Let's try to find the class in diffusers.models
    import diffusers.models
    
    found = False
    for name, obj in inspect.getmembers(diffusers.models):
        if "ZImage" in name and "Transformer" in name:
            print(f"Found class: {name}")
            sig = inspect.signature(obj.forward)
            print(f"Signature of {name}.forward:")
            print(sig)
            found = True
            
    if not found:
        print("Could not find ZImageTransformer class in diffusers.models directly.")
        print("Trying to load pipeline to get the object...")
        # Load a small dummy or just the class if we can
        # Let's try to import specifically from the likely module path
        try:
            from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
            print("Found ZImageTransformer2DModel in diffusers.models.transformers.transformer_z_image")
            sig = inspect.signature(ZImageTransformer2DModel.forward)
            print(f"Signature of ZImageTransformer2DModel.forward:")
            print(sig)
        except ImportError:
            print("Could not import from diffusers.models.transformers.transformer_z_image")

except Exception as e:
    print(f"Error: {e}")
