# Knowledge Transfer - Z-Image & ComfyUI Integration

## Essential Reference Documentation

### Z-Image Pipeline (Diffusers)

**Official ZImagePipeline Implementation**
- URL: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py
- Purpose: Understanding how Z-Image works internally in the diffusers library
- Key Information:
  - Transformer signature: `transformer(x, t, cap_feats)` (not standard diffusers signature)
  - Timestep batching requirements
  - Latent processing flow
  - How `latents` parameter is handled when passed to `__call__`

### ComfyUI Z-Image Support

**Z-Image Model Configuration in ComfyUI**
- URL: https://github.com/comfyanonymous/ComfyUI/blob/52e778fff3c1d6f32c8d14cba9864faddba8475d/comfy/supported_models.py#L1017
- Purpose: Understanding how ComfyUI natively handles Z-Image models
- Key Information:
  - Model detection and configuration
  - Native ComfyUI integration patterns
  - Differences from our diffusers-based approach

### ComfyUI ↔ Diffusers Conversion

**Diffusers State Dict Conversion**
- URL: https://github.com/comfyanonymous/ComfyUI/blob/52e778fff3c1d6f32c8d14cba9864faddba8475d/comfy/diffusers_convert.py
- Purpose: Understanding state dict conversion between ComfyUI and diffusers formats
- Key Information:
  - Weight key mapping
  - Format conversions
  - Compatibility layer implementation

## Use Cases

### When to Reference These Docs

1. **pipeline_z_image.py**
   - Debugging pipeline call issues
   - Understanding Z-Image's unique architecture
   - Implementing advanced features (e.g., custom schedulers)

2. **supported_models.py**
   - Comparing native ComfyUI vs diffusers approach
   - Understanding model loading patterns
   - Potential migration to native ComfyUI format

3. **diffusers_convert.py**
   - Converting between diffusers and ComfyUI checkpoints
   - Understanding weight compatibility
   - Troubleshooting model loading issues

## Current Implementation Notes

Our SDNQ nodes use the **diffusers library directly** rather than ComfyUI's native model handling. This approach:
- ✅ Simpler for using diffusers-specific features (SDNQ quantization)
- ✅ Access to official pipeline updates
- ⚠️ Requires understanding diffusers conventions
- ⚠️ Less integrated with ComfyUI's model management

## Related Documentation

- [Z-Image Hugging Face Model Card](https://huggingface.co/Z-a-o/Z-Image-Turbo)
- [Diffusers API Docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview)
- [ComfyUI Custom Nodes Guide](https://github.com/comfyanonymous/ComfyUI/wiki/How-to-create-a-custom-node)
