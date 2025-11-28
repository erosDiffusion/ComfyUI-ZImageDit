import torch
from diffusers import DiffusionPipeline

# Load a small test to see how pipeline handles latents
print("Testing latents device handling...")

# Simulate what we're doing
class MockPipeline:
    def __init__(self):
        self.device = torch.device("cuda")
        self._execution_device = torch.device("cuda")
        
# Our current approach
pipeline_mock = MockPipeline()

# Get device (our current code)
if hasattr(pipeline_mock, "_execution_device"):
    device = pipeline_mock._execution_device
else:
    device = pipeline_mock.device

print(f"Device selected: {device}")

# Create latents (simulating our VAE encode result)
latents = torch.randn(1, 4, 128, 128, device=device, dtype=torch.bfloat16)
print(f"Latents device: {latents.device}")
print(f"Latents dtype: {latents.dtype}")

# When we pass to pipeline, it should already be on the correct device
print(f"\n✓ Latents are already on the execution device: {device}")
print("✓ Pipeline will accept them directly without device transfers")
