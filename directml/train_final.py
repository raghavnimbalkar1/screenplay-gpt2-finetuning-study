#!/usr/bin/env python
"""
FINAL Optimized Training Pipeline (DirectML-Native, No Concatenation Packing)
Purpose: Train GPT-2 on screenplay data with maximum stability
Key Features:
  1. Custom DML-Compatible AdamW (no aten::lerp.Scalar_out CPU fallback)
  2. Batch size 1 (safe on 12GB VRAM)
  3. Gradient accumulation 8 (stable updates, effective batch=8)
  4. NO device transfers during evaluation (stays on DirectML)
  5. Original 512-token sequences (proven stable, no context breaking)
  6. Checkpoint/Resume for failure recovery
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
import sys

print("\n" + "="*80)
print("FINAL OPTIMIZED TRAINING PIPELINE (DirectML-Native)")
print("="*80 + "\n")

# ============================================
# CONFIGURATION
# ============================================
DEVICE = torch_directml.device()
MODEL_NAME = "gpt2"
DATA_DIR = Path("data/tokenized")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
RESUME_STATE_FILE = CHECKPOINT_DIR / "resume_state.json"

# Training hyperparameters - PROVEN STABLE
BATCH_SIZE = 1  # Safe for 12GB VRAM (attention is O(n^2))
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 8
EFFECTIVE_BATCH = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EVAL_STEPS = 100  # Evaluation every 100 steps
CHECKPOINT_FREQ = 500  # Checkpoint every 500 steps
MAX_SEQ_LENGTH = 512  # Original sequences (proven stable)

CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("CONFIGURATION:")
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_NAME}")
print(f"  Batch size: {BATCH_SIZE} (safe on 12GB VRAM)")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {EFFECTIVE_BATCH}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH} tokens")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Eval steps: {EVAL_STEPS}")
print(f"  Checkpoint frequency: {CHECKPOINT_FREQ} optimizer steps")
print(f"  Data dir: {DATA_DIR}\n")

# ============================================
# CUSTOM ADAMW (DML-COMPATIBLE)
# ============================================
"""
Standard AdamW uses aten::lerp.Scalar_out which is NOT supported on DirectML.
This custom implementation uses only DML-supported operations.
"""

class DMLCompatibleAdamW(torch.optim.Optimizer):
    """AdamW optimizer for DirectML (avoids unsupported operations)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Use DML-compatible operations only
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                denom = (exp_avg_sq.sqrt() + group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

# ============================================
# DATA LOADING
# ============================================
print("LOADING DATASET...")

class ScreenplayDataset(Dataset):
    """Dataset for 512-token screenplay sequences"""
    def __init__(self, tokens):
        self.tokens = tokens
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long)

train_path = DATA_DIR / "train_tokens.json"
val_path = DATA_DIR / "val_tokens.json"

if not train_path.exists():
    print(f"ERROR: {train_path} not found!")
    exit(1)

print(f"  Loading train data...")
with open(train_path, "r") as f:
    train_tokens = json.load(f)

print(f"  Loading val data...")
with open(val_path, "r") as f:
    val_tokens = json.load(f)

print(f"  Train sequences: {len(train_tokens)}")
print(f"  Val sequences: {len(val_tokens)}")

train_dataset = ScreenplayDataset(train_tokens)
val_dataset = ScreenplayDataset(val_tokens)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"  Train batches per epoch: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}\n")

# ============================================
# MODEL SETUP
# ============================================
print("LOADING MODEL...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME} ({total_params/1e6:.1f}M parameters)")
print(f"Gradient checkpointing: ENABLED")
print(f"Model device: {next(model.parameters()).device}")
print(f"No device transfers during training (stays on DirectML)\n")

# ============================================
# TRAINING SETUP
# ============================================
print("SETTING UP TRAINING...")

optimizer = DMLCompatibleAdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

print(f"Optimizer: DMLCompatibleAdamW (no CPU fallback)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {EFFECTIVE_BATCH}\n")

# ============================================
# CHECKPOINT/RESUME
# ============================================

def save_resume_state(epoch, batch_idx, global_step, best_val_loss):
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    with open(RESUME_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_resume_state():
    if RESUME_STATE_FILE.exists():
        with open(RESUME_STATE_FILE, "r") as f:
            state = json.load(f)
        print(f"[RESUME] Found state: Epoch {state['epoch']}, Step {state['global_step']}")
        return state
    return None

def load_checkpoint(checkpoint_path, model, optimizer):
    if not (checkpoint_path / "pytorch_model.bin").exists():
        return None
    checkpoint = torch.load(checkpoint_path / "pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

# Check for resume
resume_state = load_resume_state()
start_epoch = 0
start_batch_idx = 0
global_step = 0
best_val_loss = float('inf')

if resume_state:
    start_epoch = resume_state["epoch"]
    start_batch_idx = resume_state["batch_idx"]
    global_step = resume_state["global_step"]
    best_val_loss = resume_state["best_val_loss"]
    
    last_checkpoint = CHECKPOINT_DIR / f"checkpoint-{global_step}"
    if last_checkpoint.exists():
        load_checkpoint(last_checkpoint, model, optimizer)
    
    print(f"[RESUME] Resuming from epoch {start_epoch}, step {global_step}\n")

# ============================================
# TRAINING LOOP
# ============================================
print("STARTING TRAINING...")
print("-" * 80 + "\n")

accumulated_loss = 0
train_loss = 0
train_steps = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 80)
    
    model.train()
    batch_idx = start_batch_idx if epoch == start_epoch else 0
    pbar = tqdm(train_loader, desc="Training", leave=True)
    
    for batch_num, batch in enumerate(pbar):
        if batch_num < batch_idx:
            continue
        
        batch = batch.to(DEVICE)
        
        try:
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            accumulated_loss += loss.item()
            loss.backward()
            
            if (batch_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += accumulated_loss
                train_steps += 1
                accumulated_loss = 0
                global_step += 1
                
                loss_val = train_loss / train_steps
                pbar.set_postfix({'loss': f'{loss_val:.4f}', 'step': global_step})
                
                # Checkpoint
                if global_step % CHECKPOINT_FREQ == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_num,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path / "pytorch_model.bin")
                    save_resume_state(epoch, batch_num, global_step, best_val_loss)
                    print(f"\n✓ Checkpoint saved at step {global_step}")
                
                # Evaluation (NO DEVICE TRANSFERS)
                if global_step % EVAL_STEPS == 0:
                    print(f"\n[EVAL] Step {global_step}...")
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
                    print(f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = OUTPUT_DIR / "best_model"
                        best_path.mkdir(exist_ok=True)
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"✓ Best model saved (loss={avg_val_loss:.4f})")
                        save_resume_state(epoch, batch_num, global_step, best_val_loss)
                    
                    model.train()
                
                gc.collect()
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠ OOM, skipping batch...")
                optimizer.zero_grad()
                accumulated_loss = 0
            else:
                raise
    
    start_batch_idx = 0
    if train_steps > 0:
        print(f"\n✓ Epoch {epoch + 1} complete - Loss: {train_loss / train_steps:.4f}")

# ============================================
# FINAL SAVE
# ============================================
print("\n" + "-" * 80)
print("✓ TRAINING COMPLETE!")
print("-" * 80 + "\n")

final_path = OUTPUT_DIR / "final_model"
final_path.mkdir(exist_ok=True)
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

summary = {
    "epochs": NUM_EPOCHS,
    "global_steps": global_step,
    "final_train_loss": train_loss / train_steps if train_steps > 0 else 0,
    "best_val_loss": best_val_loss,
    "model_size_mb": total_params / 1e6,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH,
    "learning_rate": LEARNING_RATE,
    "optimizer": "DMLCompatibleAdamW",
    "sequence_length": MAX_SEQ_LENGTH,
    "total_sequences": len(train_tokens),
    "device": "DirectML (privateuseone:0)",
}

with open(OUTPUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total steps: {global_step}")
print(f"Final train loss: {summary['final_train_loss']:.4f}")
print(f"Best val loss: {summary['best_val_loss']:.4f}")
print(f"Model: {summary['model_size_mb']:.1f}M parameters")
print(f"Batch config: size={summary['batch_size']}, accumulation={summary['gradient_accumulation']}, effective={summary['effective_batch_size']}")
print(f"Sequences: {summary['total_sequences']} × {summary['sequence_length']} tokens")
print(f"Optimizer: {summary['optimizer']} (DML-native, no CPU fallback)")
print(f"Models saved to: {OUTPUT_DIR}/")
print("="*80)

if RESUME_STATE_FILE.exists():
    RESUME_STATE_FILE.unlink()
    print("\nResume state cleared (training completed successfully)")
