ComfyUI-ZImageDit
Alpha repo, unofficial diffusers integration of the official sdqn pipeline to run in ComfyUI

- use only sdnq nodes wich you might have to install manually , the other stuf is experimental and does not work
- install_sdnq.bat might help on windows because it looks like their toml file has an issue with license (open inside the bat and change paths)
- for flash attention find a whl, i did not bother yet
- you might need to install to latest via git to support the pipeline (from the embedded python folder):
python.exe -m pip install git+https://github.com/huggingface/diffusers.git
- 
- check requirements for what is needed (quanto is not needed but you might have trouble as there are multiple nodes here)
- weights are downloaded by diffusers on first run for sdnq nodes

- some option dont work or i did not finish porting, test.

enjoy
Enrico aka ErosDiffusion

you might have issues installing, but I have no time to support :D

note: this does not use memory management from comfy, so use carefully. memory footprint is around 7gb vram more or less, you can safely run up to 2048x2048

