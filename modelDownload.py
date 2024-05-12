from huggingface_hub import hf_hub_download
from pathlib import Path

'''
file to download models
'''

Models = [("TheBloke/Starling-LM-7B-alpha-GGUF", "Starling-LM-7B-beta-Q4_K_M.gguf"),
          ("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"),
          ("TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf")]

## Getting local directory path
local_dir = Path("Models")

## Downloading Models
for model in Models:
    hf_hub_download(model[0], filename=model[1], local_dir=local_dir)
