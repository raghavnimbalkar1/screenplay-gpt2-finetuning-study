import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from datasets import DatasetDict, Dataset
import json
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Setup Paths (Update these to your AMD machine's local paths)
DATA_DIR = "./data/tokenized" 
OUTPUT_DIR = "./03_amd_rocm_failure_study/output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 2. Data Loading (Same exact structural loader as your Mac/Cloud)
def load_tokenized_json(file_name):
    full_path = os.path.join(DATA_DIR, file_name)
    print(f"--> Loading {file_name}...")
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    payload = data["input_ids"] if isinstance(data, dict) and "input_ids" in data else data
    payload = [payload] if isinstance(payload, list) and isinstance(payload[0], int) else payload
    return Dataset.from_dict({"input_ids": payload})

raw_datasets = DatasetDict({
    "train": load_tokenized_json("train_tokens.json"),
    "validation": load_tokenized_json("val_tokens.json")
})

# Context chunking
block_size = 512
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = raw_datasets.map(group_texts, batched=True, num_proc=1)

# 3. Model & Tokenizer Initialization
print("--> Initializing base GPT-2 model on AMD ROCm...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model directly to the ROCm backend
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

# 4. Inject LoRA Adapters
print("--> Injecting LoRA adapters to bypass AMD VRAM limitations...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1,
    target_modules=["c_attn"] # Targeting GPT-2 attention blocks
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training Configuration (AMD Optimized)
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    learning_rate=3e-4,              # Slightly higher LR for LoRA adapters
    per_device_train_batch_size=4,   # Keeping batch size small to dodge the OOM crash
    gradient_accumulation_steps=1,   # Removed accumulation to prevent backend graph caching crashes
    fp16=True,                       # AMD Hardware Half-Precision (Crucial VRAM saver)
    logging_steps=50,
    max_steps=4700,                  # Matching the Mac's workload limit for a 1:1 hardware comparison
    dataloader_num_workers=0,        # Preventing multi-thread memory leaks
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

# 6. Execute Training
print("\n--> Commencing AMD ROCm LoRA Fine-Tuning Sequence...")
trainer.train()

print("\n--> Training Complete. Saving AMD LoRA adapters...")
final_model_path = os.path.join(OUTPUT_DIR, "final_amd_lora")
trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

import pandas as pd
pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(OUTPUT_DIR, "amd_lora_benchmarks.csv"), index=False)
print("--> Success! Adapters and telemetry saved.")