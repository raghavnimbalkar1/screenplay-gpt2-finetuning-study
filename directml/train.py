#!/usr/bin/env python
"""
Full-Scale Training Pipeline for GPT-2 on Screenplay Data
Purpose: Train GPT-2 on entire screenplay dataset with memory optimization for AMD RX 6700XT
Device: AMD RX 6700XT (12GB VRAM) with DirectML backend
"""

import os
import json
import torch
import torch_directml
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc

print("\n" + "="*70)
print("FULL-SCALE TRAINING PIPELINE (DIRECTML + AMD GPU)")
print("="*70 + "\n")

# ============================================
# CONFIGURATION
# ============================================
DEVICE = torch_directml.device()
MODEL_NAME = "gpt2"
DATA_DIR = Path("data/tokenized")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

# Training hyperparameters (conservative for 12GB VRAM)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EVAL_STEPS = 50
SAVE_STEPS = 100
WARMUP_STEPS = 100
# NO MAX_TRAIN_BATCHES - uses full dataset

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("CONFIGURATION:")
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_NAME}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Data dir: {DATA_DIR}\n")

# ============================================
# DATA LOADING
# ============================================
print("LOADING DATASET...")

class ScreenplayDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long)

# Load tokenized data
train_path = DATA_DIR / "train_tokens.json"
val_path = DATA_DIR / "val_tokens.json"

if not train_path.exists():
    print("ERROR: Tokenized data not found!")
    print(f"Expected: {train_path}")
    exit(1)

print("  Loading train data...")
with open(train_path, "r") as f:
    train_tokens = json.load(f)

print("  Loading val data...")
with open(val_path, "r") as f:
    val_tokens = json.load(f)

# Use FULL dataset - no slicing
train_dataset = ScreenplayDataset(train_tokens)
val_dataset = ScreenplayDataset(val_tokens)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}\n")

# ============================================
# MODEL SETUP
# ============================================
print("LOADING MODEL...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME} ({total_params/1e6:.1f}M parameters)")
print(f"Gradient checkpointing: ENABLED")
print(f"Model device: {next(model.parameters()).device}\n")

# ============================================
# TRAINING SETUP
# ============================================
print("SETTING UP TRAINING...")

# Use AdamW with fused=False to avoid DirectML unsupported operations
# The warning 'aten::lerp.Scalar_out' is not supported on DML backend occurs with
# standard Adam. AdamW avoids this fallback to CPU during optimizer.step()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    fused=False  # Critical: fused=True would trigger unsupported DML operations
)

print(f"Optimizer: AdamW with fused=False (lr={LEARNING_RATE})")
print(f"  Note: fused=False prevents CPU fallback on DirectML backend")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps\n")

# ============================================
# TRAINING LOOP
# ============================================
print("STARTING TRAINING...")
print("-" * 70 + "\n")

global_step = 0
best_val_loss = float('inf')
accumulated_loss = 0

for epoch in range(NUM_EPOCHS):
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Training
    model.train()
    train_loss = 0
    train_steps = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # No MAX_TRAIN_BATCHES limit - process entire dataset
        batch = batch.to(DEVICE)
        
        # Forward pass
        try:
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            accumulated_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Optimizer step every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += accumulated_loss
                train_steps += 1
                accumulated_loss = 0
                global_step += 1
                
                loss_val = train_loss / train_steps if train_steps > 0 else 0
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
                
                # Periodic evaluation
                if global_step % EVAL_STEPS == 0:
                    model.eval()
                    val_loss = 0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = val_batch.to(DEVICE)
                            outputs = model(val_batch, labels=val_batch)
                            val_loss += outputs.loss.item()
                            val_steps += 1
                    
                    avg_val_loss = val_loss / val_steps
                    avg_train_loss = train_loss / train_steps
                    
                    print(f"\nStep {global_step}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = OUTPUT_DIR / "best_model"
                        best_path.mkdir(exist_ok=True)
                        
                        # Move model to CPU before saving to avoid opaque tensor issues
                        model_cpu = model.to("cpu")
                        model_cpu.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        model_cpu = model_cpu.to(DEVICE)
                        del model_cpu
                        
                        print(f"Best model saved: {best_path}")
                    
                    model.train()
                
                # Periodic checkpoint
                if global_step % SAVE_STEPS == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path / "pytorch_model.bin")
                    
                    print(f"Checkpoint saved: {checkpoint_path}")
                
                # Free memory
                gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nGPU out of memory at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                optimizer.zero_grad()
                accumulated_loss = 0
            else:
                raise
    
    # End of epoch
    if train_steps > 0:
        avg_train_loss = train_loss / train_steps
        print(f"\nEpoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

# ============================================
# FINAL SAVE
# ============================================
print("\n" + "-" * 70)
print("TRAINING COMPLETE!")
print("-" * 70 + "\n")

# Save final model
final_path = OUTPUT_DIR / "final_model"
final_path.mkdir(exist_ok=True)

# Move model to CPU before saving to avoid opaque tensor issues
model_cpu = model.to("cpu")
model_cpu.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
del model_cpu

print(f"Final model saved: {final_path}\n")

# Save training summary
summary = {
    "epochs": NUM_EPOCHS,
    "global_steps": global_step,
    "final_train_loss": avg_train_loss if train_steps > 0 else 0,
    "best_val_loss": best_val_loss,
    "model_size_mb": total_params / 1e6,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
}

with open(OUTPUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Total steps: {global_step}")
print(f"Total epochs: {NUM_EPOCHS}")
if train_steps > 0:
    print(f"Final train loss: {avg_train_loss:.4f}")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Model saved to: {final_path}")
print()
print("Ready for STEP 6: Generate screenplay samples!")
print()
