import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Define paths (Use your local folder or HF repo name)
base_model_id = "gpt2"
adapter_path = "./models/mac_lora_adapters/final_adapters" # Or "YOUR_USERNAME/gpt2-screenplay-mac-lora"

print("--> Loading Base GPT-2 Model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# Ensure padding token is set, matching your training config
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to("mps")

print("--> Injecting LoRA Adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path).to("mps")

# 2. Setup the prompt
prompt = """EXT. CYBERPUNK ALLEYWAY - NIGHT

Neon lights flicker in the puddles. A lone DETECTIVE (40s) lights a cigarette, looking at the broken android on the ground.

DETECTIVE"""

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

print("--> Generating Scene...\n")
print("="*50)

# 3. Generate text using optimal nucleus sampling
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.85,          # Slightly higher for creative variance
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.15,   # Crucial to stop looping dialogue
        pad_token_id=tokenizer.pad_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*50)