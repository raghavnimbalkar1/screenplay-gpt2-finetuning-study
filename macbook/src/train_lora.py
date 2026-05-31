import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model

# 1. Path Configurations
DATA_DIR = "./data"
MODEL_OUTPUT_DIR = "./models/mac_lora_adapters"
LOG_DIR = "./logs/mac_training_logs"

print("--> Loading pre-tokenized dataset...")
raw_datasets = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "train_tokens.json"),
        "validation": os.path.join(DATA_DIR, "val_tokens.json")
    }
)

print("--> Healing dataset structure: Merging matrix columns into input_ids...")
def heal_dataset(batch):
    # Dynamically find all stringified column keys from '0' to '511'
    cols = [str(i) for i in range(512) if str(i) in batch]
    # High-speed python zip to reconstruct rows of token lists
    input_ids = [list(row) for row in zip(*(batch[k] for k in cols))]
    return {"input_ids": input_ids, "labels": input_ids}

# Map the healer over the splits, stripping out the old 512 numerical columns
datasets = raw_datasets.map(
    heal_dataset,
    batched=True,
    batch_size=1000,
    remove_columns=raw_datasets["train"].column_names
)

# --- RESEARCH QUALITY WORKLOAD SUBSAMPLED BOUNDARY ---
print("--> Optimizing validation set size for MacBook hardware constraints...")
# Capping validation at 2,000 entries gives high statistical stability without intermediate freezes
datasets["validation"] = datasets["validation"].select(range(2000))


# 2. Hardware and Model Initialization
print("--> Initializing base GPT-2 model on MPS (Metal Performance Shaders)...")
base_model = GPT2LMHeadModel.from_pretrained("gpt2")


# 3. LoRA (Parameter-Efficient Fine-Tuning) Configuration
print("--> Injecting LoRA adapters into attention layers...")
lora_config = LoraConfig(
    r=8,                           # Rank: How wide the adapter matrices are
    lora_alpha=32,                 # Scaling factor
    target_modules=["c_attn"],     # Targets GPT-2's attention projection layers
    lora_dropout=0.05,             # Dropout to prevent overfitting on the adapters
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the base model to freeze 99% of weights and only train the adapters
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()


# 3.5 Load Tokenizer for the Data Collator
print("--> Loading GPT-2 tokenizer configuration...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to prevent GPT-2 alignment bugs


# 4. Data Collator (Now equipped with the tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# 5. Training Arguments (Rigorously tuned for 8-9 Hour Mac Research Run)
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=1000,                 # CRITICAL: Dropped frequency to prevent intermediate freezes
    save_strategy="steps",
    save_steps=1000,                 # Sync checkpoints directly with evaluation cadence
    save_total_limit=2,              # Keep two historic checkpoints to save disk memory space
    logging_dir=LOG_DIR,
    logging_steps=50,                # Clean terminal logs every 50 steps
    learning_rate=3e-4,              
    weight_decay=0.01,
    
    # --- MEMORY DROPPER PROTOCOLS ---
    per_device_train_batch_size=1,   # Drops batch sizes to prevent swap-to-disk leaks
    per_device_eval_batch_size=1,    
    gradient_accumulation_steps=16,  # Accumulates 16 steps sequentially for stable optimization
    dataloader_num_workers=0,        # Single process data-loading prevents threading locks
    # ---------------------------------
    
    # --- BALANCED COMPUTE BUDGET ---
    max_steps=4700,                  # Exactly ~8.5 hours of solid execution (cruncing ~75k entries)
    fp16=False,                      # Explicit native precision mapping for Apple Silicon
    report_to="none"
)


# 6. Initialize Trainer and Launch
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,             # Seamless serialization integration
)

print("--> Commencing Apple Silicon LoRA Fine-Tuning Sequence...")
trainer.train()


# 7. Save the Final Adapters
print("--> Training Complete. Saving LoRA adapters...")
trainer.save_model(os.path.join(MODEL_OUTPUT_DIR, "final_adapters"))
print("--> Run finished successfully!")