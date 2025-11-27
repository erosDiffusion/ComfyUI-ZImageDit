from .loader_node import LoadZImageTurboQDiTOffline
from .generator_node import ZImageTurboQDiTGenerateUnload
from .lightweight_nodes import LoadQuantoZImageTransformer, LoadQuantoZImageCLIP, LoadQuantoZImageVAE, ZImageQuantoSampler
from .comfy_quant_nodes import QuantizeActiveModel, QuantizeComfyModelHybrid
from .sdnq_nodes import LoadZImageSDNQ, ZImageSDNQGenerate

NODE_CLASS_MAPPINGS = {
    "LoadZImageTurboQDiTOffline": LoadZImageTurboQDiTOffline,
    "ZImageTurboQDiTGenerateUnload": ZImageTurboQDiTGenerateUnload,
    "LoadQuantoZImageTransformer": LoadQuantoZImageTransformer,
    "LoadQuantoZImageCLIP": LoadQuantoZImageCLIP,
    "LoadQuantoZImageVAE": LoadQuantoZImageVAE,
    "ZImageQuantoSampler": ZImageQuantoSampler,
    "QuantizeActiveModel": QuantizeActiveModel,
    "QuantizeComfyModelHybrid": QuantizeComfyModelHybrid,
    "LoadZImageSDNQ": LoadZImageSDNQ,
    "ZImageSDNQGenerate": ZImageSDNQGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadZImageTurboQDiTOffline": "Load Z-Image Turbo (Q-DiT)",
    "ZImageTurboQDiTGenerateUnload": "Z-Image Turbo Generate (Unload)",
    "LoadQuantoZImageTransformer": "Load Z-Image Transformer (Quanto)",
    "LoadQuantoZImageCLIP": "Load Z-Image CLIP (Quanto)",
    "LoadQuantoZImageVAE": "Load Z-Image VAE",
    "ZImageQuantoSampler": "Z-Image Quanto Sampler",
    "QuantizeActiveModel": "Quantize Active Model (Quanto)",
    "QuantizeComfyModelHybrid": "Quantize Model Hybrid (Quanto)",
    "LoadZImageSDNQ": "Load Z-Image Turbo (SDNQ)",
    "ZImageSDNQGenerate": "Z-Image Generate (SDNQ)",
}
