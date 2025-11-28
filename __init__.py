from .sdnq_nodes import LoadZImageSDNQ, ZImageSDNQGenerate

NODE_CLASS_MAPPINGS = {
    "LoadZImageSDNQ": LoadZImageSDNQ,
    "ZImageSDNQGenerate": ZImageSDNQGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadZImageSDNQ": "Load Z-Image Turbo (SDNQ)",
    "ZImageSDNQGenerate": "Z-Image Generate (SDNQ)",
}
