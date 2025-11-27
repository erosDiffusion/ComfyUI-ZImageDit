# ComfyUI-ZImageDit
<img width="1024" height="1024" alt="ComfyUI-zimage-diffusers-wrapper_00002_" src="https://github.com/user-attachments/assets/eac45a4d-ad75-4a0f-9de7-b6bcf51f2292" />

<img width="2139" height="1075" alt="image" src="https://github.com/user-attachments/assets/bac94951-c82c-46f4-ab14-089875693072" />


## What is this ?
- an Alpha repo: unofficial **diffusers** integration of the official **SDNQ pipeline** to run in ComfyUI ...because i wanted to compare quality and be even more vram savy

## Notes:
- **use only sdnq nodes** wich you might have to install manually , the other stuf is experimental and does not work
- **install_sdnq.bat** might help on windows because it looks like their toml file has an issue with license (open inside the bat and change paths)
- for flash attention find a whl, I did not bother yet as it's ok-ish speedwise, if you need you can try these places:
  - prebuilt wheels  https://huggingface.co/Kijai/PrecompiledWheels/tree/main
  - prebuilt wheels https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.4.10
  - prebuilt wheels https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main
- compile does not work
- you might need to **install the latest diffusers manually via git to support the pipeline** (from the embedded python folder):
python.exe -m pip install git+https://github.com/huggingface/diffusers.git
- check requirements for what is needed (quanto is not needed but you might have trouble as there are multiple nodes here)
- weights are downloaded by diffusers on first run for sdnq nodes
- some option dont work or i did not finish porting, test.
- only tested on windows
  - Platform: Windows
  - Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
  - pytorch version: 2.8.0+cu128
  - xformers version: 0.0.32.post2
  - Set vram state to: NORMAL_VRAM
  - Device: cuda:0 NVIDIA GeForce RTX 3080 : cudaMallocAsync
  - ComfyUI version: 0.3.75
  - ComfyUI frontend version: 1.33.8
  - Total VRAM 10240 MB, total RAM 32560 MB


- if you are on linux you are smart enought to know what to do

Enjoy!
Enrico aka ErosDiffusion

ps.: you might have issues installing, but I have no time to support :D

note: this does not use memory management from comfy, so use carefully. memory footprint is around 7gb vram more or less, you can safely run up to 2048x2048

