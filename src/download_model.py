import os
from huggingface_hub import hf_hub_download

# This ensures we don't time out on your HDD
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# Path to your models folder
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "..", "models")

# We are only downloading the 4-bit version (Smallest & Fastest)
files_to_download = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model_q4f16.onnx" # This is the ONLY brain file you need
]

print(f"--- Starting Targeted Download to {model_dir} ---")

for file in files_to_download:
    print(f"Fetching {file}...")
    hf_hub_download(
        repo_id="onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
        filename=file,
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )

print("\n✅ SUCCESS! You have the 'model_q4f16.onnx' file and tokenizers.")