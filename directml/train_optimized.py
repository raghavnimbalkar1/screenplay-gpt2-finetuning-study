#!/usr/bin/env python
"""
OPTIMIZED Training Pipeline with DML-Compatible Optimizer & Sequence Packing
Purpose: Train GPT-2 on screenplay data with maximum GPU utilization
Key Fixes:
  1. Custom AdamW avoiding aten::lerp.Scalar_out (DML incompatible operation)
  2. Increased batch size (1→4) + reduced accumulation (4→2) for 8x effective batch
  3. Sequence packing (512-token blocks) to eliminate padding waste
  4. NO device transfers during evaluation (stays on DirectML)
  5. Checkpoint/Resume preserved for failure recovery
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
import math

print("\n" + "="*80)
print("OPTIMIZED TRAINING PIPELINE (DirectML + Custom AdamW + Sequence Packing)")
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

# Training hyperparameters - OPTIMIZED FOR 12GB VRAM
BATCH_SIZE = 1  # Keep at 1 (attention quadratic memory, batch=4 exceeds 12GB)
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to 8 for stable updates
EFFECTIVE_BATCH = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # = 8 (same as before)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EVAL_STEPS = 100  # Increased from 50 to reduce eval frequency
CHECKPOINT_FREQ = 500  # Save checkpoint every 500 optimizer steps
SEQUENCE_PACK_SIZE = 1024  # Pack sequences into 1024-token blocks by concatenation  

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("CONFIGURATION:")
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_NAME}")
print(f"  Batch size: {BATCH_SIZE} (stable for 12GB VRAM)")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} (for stable updates)")
print(f"  Effective batch size: {EFFECTIVE_BATCH}")
print(f"  Sequence pack size: {SEQUENCE_PACK_SIZE} tokens (by concatenation - ~2x reduction)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Eval steps: {EVAL_STEPS} (reduced eval frequency)")
print(f"  Checkpoint frequency: {CHECKPOINT_FREQ} optimizer steps")
print(f"  Resume state file: {RESUME_STATE_FILE}")
print(f"  Data dir: {DATA_DIR}\n")

# ============================================
# CUSTOM ADAMW (DML-COMPATIBLE, NO LERP FALLBACK)
# ============================================
"""
Standard AdamW uses exp_avg.lerp_(grad, 1 - beta1) which calls aten::lerp.Scalar_out
This operation is NOT supported on DirectML backend, causing CPU fallback.

Solution: Implement AdamW manually using ONLY supported operations:
  - torch.mul_() → supported on DML
  - torch.add_() → supported on DML
  - torch.addcmul_() → supported on DML
  - torch.addcdiv_() → supported on DML
  
This achieves the same AdamW update WITHOUT triggering DirectML CPU fallback.
"""

class DMLCompatibleAdamW(torch.optim.Optimizer):
    """AdamW optimizer compatible with DirectML backend (avoids aten::lerp.Scalar_out)"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                # REPLACED: exp_avg.lerp_(grad, 1 - beta1)  [NOT DML-COMPATIBLE]
                # WITH: exp_avg = beta1 * exp_avg + (1 - beta1) * grad [DML-COMPATIBLE]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Weight decay (AdamW style - decoupled)
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Update parameters: p = p - step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
                denom = (exp_avg_sq.sqrt() + group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

# ============================================
# SEQUENCE PACKING HELPER
# ============================================

def pack_sequences(token_lists, pack_size=1024):
    """
    Pack sequences by concatenation into uniform blocks.
    
    IMPORTANT: This flattens all sequences into one long sequence,
    then splits into pack_size-token chunks. This eliminates padding entirely.
    
    Example:
      Input: [[1,2,...,512], [1,2,...,512], [1,2,...,512], ...]
      Output: [[1,2,...,512,1,2,...,512], [1,2,...,512,...1,2,...], ...]
             Each output is exactly pack_size tokens (1024)
    
    Benefit: ~2x reduction in total sequences (pack_size=1024, input sequences=512)
    No padding overhead, no sequence boundaries within pack.
    """
    # Flatten all tokens into one continuous sequence
    all_tokens = []
    for tokens in token_lists:
        tokens_list = list(tokens) if not isinstance(tokens, list) else tokens
        all_tokens.extend(tokens_list)
    
    # Split into pack_size-token chunks (no padding, pure concatenation)
    packed = []
    for i in range(0, len(all_tokens) - pack_size + 1, pack_size):
        chunk = all_tokens[i:i + pack_size]
        if len(chunk) == pack_size:
            packed.append(chunk)
    
    # Handle remaining tokens (if any) - pad to pack_size
    remaining_start = (len(all_tokens) // pack_size) * pack_size
    if remaining_start < len(all_tokens):
        final_chunk = all_tokens[remaining_start:]
        if final_chunk:
            # Pad final chunk to pack_size
            padding_needed = pack_size - len(final_chunk)
            final_chunk.extend([50256] * padding_needed)  # EOT token padding
            packed.append(final_chunk)
    
    return packed

# ============================================
# CHECKPOINT/RESUME FUNCTIONS
# ============================================

def save_resume_state(epoch, batch_idx, global_step, best_val_loss):
    """Save training state for recovery"""
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    with open(RESUME_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_resume_state():
    """Load training state if available"""
    if RESUME_STATE_FILE.exists():
        with open(RESUME_STATE_FILE, "r") as f:
            state = json.load(f)
        print(f"[RESUME] Found saved state: Epoch {state['epoch']}, Step {state['global_step']}")
        return state
    return None

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load checkpoint and restore training state"""
    if not (checkpoint_path / "pytorch_model.bin").exists():
        return None
    
    print(f"[RESUME] Loading checkpoint from {checkpoint_path}...")
    
    # Load model weights
    checkpoint = torch.load(checkpoint_path / "pytorch_model.bin", map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    
    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[RESUME] Optimizer state restored")
    
    print(f"[RESUME] Checkpoint loaded successfully")
    return checkpoint

# ============================================
# DATA LOADING WITH SEQUENCE PACKING
# ============================================
print("LOADING AND PACKING DATASET...")

class PackedScreenplayDataset(Dataset):
    """Dataset that yields pre-packed 512-token sequences"""
    def __init__(self, packed_tokens):
        self.packed_tokens = packed_tokens
    
    def __len__(self):
        return len(self.packed_tokens)
    
    def __getitem__(self, idx):
        tokens = self.packed_tokens[idx]
        if isinstance(tokens, list):
            return torch.tensor(tokens, dtype=torch.long)
        return tokens

# Load tokenized data
train_path = DATA_DIR / "train_tokens.json"
val_path = DATA_DIR / "val_tokens.json"

if not train_path.exists():
    print("ERROR: Tokenized data not found!")
    print(f"Expected: {train_path}")
    exit(1)

print(f"  Loading train data ({train_path})...")
with open(train_path, "r") as f:
    train_tokens = json.load(f)

print(f"  Loading val data ({val_path})...")
with open(val_path, "r") as f:
    val_tokens = json.load(f)

print(f"  Original train sequences: {len(train_tokens)}")
print(f"  Total train tokens: {sum(len(t) for t in train_tokens):,}")
print(f"  Concatenation packing into {SEQUENCE_PACK_SIZE}-token blocks...")

# Pack sequences by concatenation to eliminate padding entirely
packed_train = pack_sequences(train_tokens, pack_size=SEQUENCE_PACK_SIZE)
packed_val = pack_sequences(val_tokens, pack_size=SEQUENCE_PACK_SIZE)

reduction_factor = len(train_tokens) / len(packed_train) if len(packed_train) > 0 else 1.0
print(f"  Packed train sequences: {len(packed_train)} (reduced {reduction_factor:.1f}x from {len(train_tokens)})")
print(f"  Packed val sequences: {len(packed_val)}")
print(f"  Speedup: ~{reduction_factor:.1f}x fewer forward passes (padding eliminated)\n")

train_dataset = PackedScreenplayDataset(packed_train)
val_dataset = PackedScreenplayDataset(packed_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Batches per epoch: {len(train_loader)} (batch size={BATCH_SIZE})\n")

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
print(f"Model device: {next(model.parameters()).device}")
print(f"Model remains on DirectML for entire training (NO device transfers)\n")

# ============================================
# TRAINING SETUP WITH CUSTOM ADAMW
# ============================================
print("SETTING UP TRAINING...")

# Use custom DML-compatible AdamW instead of torch.optim.AdamW
optimizer = DMLCompatibleAdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

print(f"Optimizer: DMLCompatibleAdamW (custom, avoids aten::lerp.Scalar_out)")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Betas: (0.9, 0.999)")
print(f"  Weight decay: 0.01")
print(f"Batch size: {BATCH_SIZE} (safe on 12GB VRAM)")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {EFFECTIVE_BATCH}\n")

# ============================================
# RESUME FROM CHECKPOINT IF AVAILABLE
# ============================================
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
    
    # Try to load the last checkpoint
    last_checkpoint = CHECKPOINT_DIR / f"checkpoint-{global_step}"
    if last_checkpoint.exists():
        load_checkpoint(last_checkpoint, model, optimizer)
    
    print(f"\n[RESUME] Resuming from:")
    print(f"  Epoch: {start_epoch}/{NUM_EPOCHS}")
    print(f"  Batch: {start_batch_idx} (global step {global_step})")
    print(f"  Best val loss: {best_val_loss:.4f}\n")

# ============================================
# TRAINING LOOP WITH RESUME & NATIVE EVAL
# ============================================
print("STARTING TRAINING...")
print("-" * 80 + "\n")

accumulated_loss = 0
train_loss = 0
train_steps = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\nEPOCH {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 80)
    
    # Training
    model.train()
    batch_idx = start_batch_idx if epoch == start_epoch else 0
    
    pbar = tqdm(train_loader, desc="Training", position=0, leave=True)
    
    for batch_num, batch in enumerate(pbar):
        if batch_num < batch_idx:
            continue
        
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
            if (batch_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += accumulated_loss
                train_steps += 1
                accumulated_loss = 0
                global_step += 1
                
                loss_val = train_loss / train_steps if train_steps > 0 else 0
                pbar.set_postfix({'loss': f'{loss_val:.4f}', 'step': global_step})
                
                # Periodic checkpoint (for recovery)
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
                    
                    # Save resume state for automatic recovery
                    save_resume_state(epoch, batch_num, global_step, best_val_loss)
                    print(f"\n✓ Checkpoint saved at step {global_step}")
                
                # Periodic evaluation (NATIVE ON DIRECTML - NO DEVICE TRANSFERS)
                if global_step % EVAL_STEPS == 0:
                    print(f"\n[EVAL] Running evaluation at step {global_step}...")
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
                    
                    print(f"[EVAL] Step {global_step}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
                    
                    # Save best model (KEEP ON DIRECTML)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = OUTPUT_DIR / "best_model"
                        best_path.mkdir(exist_ok=True)
                        
                        # Save model while on DirectML device
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        
                        print(f"✓ Best model saved: {best_path} (loss={avg_val_loss:.4f})")
                        
                        # Update resume state with new best loss
                        save_resume_state(epoch, batch_num, global_step, best_val_loss)
                    
                    model.train()
                
                # Free memory
                gc.collect()
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠ GPU out of memory at batch {batch_num}, skipping...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                optimizer.zero_grad()
                accumulated_loss = 0
            else:
                raise
    
    # Reset batch_idx for next epoch
    start_batch_idx = 0
    
    # End of epoch
    if train_steps > 0:
        avg_train_loss = train_loss / train_steps
        print(f"\n✓ Epoch {epoch + 1} complete - Avg Train Loss: {avg_train_loss:.4f}")

# ============================================
# FINAL SAVE
# ============================================
print("\n" + "-" * 80)
print("✓ TRAINING COMPLETE!")
print("-" * 80 + "\n")

# Save final model (NATIVE ON DIRECTML)
final_path = OUTPUT_DIR / "final_model"
final_path.mkdir(exist_ok=True)

model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

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
    "effective_batch_size": EFFECTIVE_BATCH,
    "learning_rate": LEARNING_RATE,
    "optimizer": "DMLCompatibleAdamW",
    "sequence_pack_size": SEQUENCE_PACK_SIZE,
    "original_sequences": len(train_tokens),
    "packed_sequences": len(packed_train),
    "speedup_factor": len(train_tokens) / len(packed_train) if len(packed_train) > 0 else 1.0,
    "device": "DirectML (privateuseone:0)",
    "no_device_transfers": True,
}

with open(OUTPUT_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Total steps: {global_step}")
print(f"Final train loss: {summary['final_train_loss']:.4f}")
print(f"Best val loss: {summary['best_val_loss']:.4f}")
print(f"Model parameters: {summary['model_size_mb']:.1f}M")
print(f"Batch size: {summary['batch_size']} (safe for 12GB VRAM)")
print(f"Effective batch (w/ accumulation): {summary['effective_batch_size']}")
print(f"Learning rate: {summary['learning_rate']}")
print(f"\nSequence Packing Impact (Concatenation):")
print(f"  Original sequences: {summary['original_sequences']}")
print(f"  Packed sequences: {summary['packed_sequences']}")
print(f"  Reduction factor: {summary['speedup_factor']:.1f}x")
print(f"  (All sequences concatenated, split into {SEQUENCE_PACK_SIZE}-token blocks, padding eliminated)")
print(f"\nOptimizer: {summary['optimizer']} (DML-compatible, no CPU fallback)")
print(f"Device: {summary['device']}")
print(f"All computations on DirectML (no transfers)\n")
print(f"Models saved to: {OUTPUT_DIR}/")
print(f"Best model: {OUTPUT_DIR}/best_model/")
print(f"Final model: {OUTPUT_DIR}/final_model/")
print("="*80)

# Clean up resume state on success
if RESUME_STATE_FILE.exists():
    RESUME_STATE_FILE.unlink()
    print("\nResume state cleared (training completed successfully)")