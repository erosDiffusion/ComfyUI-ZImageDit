
import subprocess
import sys

def install():
    packages = [
        "diffusers>=0.35.2",
        "transformers>=4.56.1",
        "safetensors>=0.4.0",
        "quanto>=0.1.0"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

if __name__ == "__main__":
    install()
