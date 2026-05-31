#!/usr/bin/env python
# STEP 2: GPU VERIFICATION TEST
# Simple test of DirectML, GPT-2, and GPU functionality

import torch
import torch_directml
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("\n" + "="*70)
print("STEP 2: GPU VERIFICATION TEST")
print("="*70 + "\n")

# Test 1: DirectML
print("TEST 1: DirectML Detection")
print("-" * 70)
device = torch_directml.device()
print(f"DirectML Device: {device}")
print(f"Status: SUCCESS\n")

# Test 2: GPU Tensor Operations
print("TEST 2: GPU Tensor Operations")
print("-" * 70)
t1 = torch.randn(1000, 1000).to(device)
t2 = torch.randn(1000, 1000).to(device)
result = torch.matmul(t1, t2)
print(f"Matrix multiplication on GPU: {result.shape}")
print(f"Status: SUCCESS\n")

# Test 3: Load Model
print("TEST 3: GPT-2 Model Loading")
print("-" * 70)
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Loading model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {total_params/1e6:.1f}M parameters")
print(f"Status: SUCCESS\n")

# Test 4: Move to GPU
print("TEST 4: Move Model to GPU")
print("-" * 70)
model = model.to(device)
first_param = next(model.parameters())
print(f"Model device: {first_param.device}")
print(f"Status: SUCCESS\n")

# Test 5: Forward Pass
print("TEST 5: Forward Pass")
print("-" * 70)
text = "INT. HOUSE - NIGHT\n\nA character enters the room."
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
model.eval()
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
print(f"Loss: {outputs.loss.item():.4f}")
print(f"Logits shape: {outputs.logits.shape}")
print(f"Status: SUCCESS\n")

# Test 6: Generation
print("TEST 6: Text Generation")
print("-" * 70)
prompt = "INT. COFFEE SHOP"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(input_ids, max_length=30, temperature=0.7)
generated = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated: {generated}")
print(f"Status: SUCCESS\n")

# Summary
print("="*70)
print("ALL TESTS PASSED - GPU IS READY FOR TRAINING!")
print("="*70)
print()
print("Summary:")
print("  [OK] DirectML detected and working")
print("  [OK] GPU tensor operations functional")
print("  [OK] GPT-2 model loads successfully")
print("  [OK] Model moves to GPU correctly")
print("  [OK] Forward passes work on GPU")
print("  [OK] Text generation works end-to-end")
print()
print("Your AMD GPU is ready for fine-tuning!")
print()
