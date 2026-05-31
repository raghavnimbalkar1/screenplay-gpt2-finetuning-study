from huggingface_hub import HfApi

api = HfApi()

# Replace with your actual Hugging Face username
repo_id = "raghavnimbalkar/gpt2-screenplay-mac-lora" 

# Create the repository (it won't overwrite if it already exists)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print(f"Pushing LoRA adapters to {repo_id}...")

# Upload the folder containing your adapter_model.safetensors and adapter_config.json
api.upload_folder(
    folder_path="./models/mac_lora_adapters/final_adapters",  
    repo_id=repo_id,
    repo_type="model"
)

print("Upload complete! Your Mac model is now live on Hugging Face.")