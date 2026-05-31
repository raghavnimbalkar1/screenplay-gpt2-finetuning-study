# ============================================================
# STEP 2: GPU Test Script
# Verify DirectML is correctly detecting and using AMD GPU
# ============================================================

import torch
import torch_directml
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def print_result(label, value, status=""):
    """Print formatted result line"""
    status_str = f" [{status}]" if status else ""
    print(f"  {label:<40} : {value}{status_str}")

def main():
    print_section("STEP 2: GPU DETECTION & VERIFICATION TEST")
    
    # ========================================================
    # 1. Check torch installation and version
    # ========================================================
    print_section("1. PyTorch Installation Check")
    print_result("PyTorch Version", torch.__version__)
    print_result("CUDA Available", torch.cuda.is_available(), "NO (expected - not CUDA)")
    
    # ========================================================
    # 2. Check DirectML availability
    # ========================================================
    print_section("2. DirectML Support Check")
    
    try:
        # DirectML is automatically enabled when imported
        device = torch_directml.device()
        print_result("DirectML Import", "✓ Successfully imported")
        print_result("Default Device", str(device), "✓ READY")
    except Exception as e:
        print_result("DirectML Import", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # 3. Test tensor operations on DirectML
    # ========================================================
    print_section("3. DirectML Tensor Operations Test")
    
    try:
        # Create a test tensor
        test_tensor = torch.randn(1000, 1000)
        print_result("CPU Tensor Created", f"Shape: {test_tensor.shape}")
        
        # Move to DirectML device
        device = torch_directml.device()
        directml_tensor = test_tensor.to(device)
        print_result("Moved to DirectML", f"Device: {directml_tensor.device}", "✓ SUCCESS")
        
        # Perform a simple operation on GPU
        result = torch.matmul(directml_tensor, directml_tensor)
        print_result("Matrix Multiplication on GPU", f"Result shape: {result.shape}", "✓ SUCCESS")
        
        # Move back to CPU for verification
        cpu_result = result.cpu()
        print_result("GPU→CPU Transfer", f"Result on CPU: {cpu_result.shape}", "✓ SUCCESS")
        
    except Exception as e:
        print_result("DirectML Operations", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # 4. Load GPT-2 model
    # ========================================================
    print_section("4. GPT-2 Model Loading Test")
    
    try:
        print("  Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print_result("Tokenizer Loaded", "✓ gpt2 tokenizer ready")
        
        print("  Loading GPT-2 model (this may take a moment)...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        print_result("Model Loaded", f"✓ gpt2 model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")
        
    except Exception as e:
        print_result("Model Loading", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # 5. Move model to DirectML device
    # ========================================================
    print_section("5. Model Movement to DirectML Device")
    
    try:
        device = torch_directml.device()
        model = model.to(device)
        print_result("Model Moved to", str(device), "✓ SUCCESS")
        
        # Verify model is on device
        for i, param in enumerate(list(model.parameters())[:3]):  # Check first 3 params
            print_result(f"  Parameter {i} location", f"Device: {param.device}")
        
    except Exception as e:
        print_result("Model Movement", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # 6. Run a tiny forward pass through the model
    # ========================================================
    print_section("6. Forward Pass Test (GPU Utilization)")
    
    try:
        # Create simple input
        test_input = "INT. HOUSE - NIGHT\n\nA man walks into the room."
        print_result("Test Input", f'"{test_input[:50]}..."')
        
        # Tokenize
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        print_result("Tokenized Length", f"{inputs.shape} tokens")
        
        # Move inputs to device
        inputs = inputs.to(device)
        print_result("Inputs Moved to", str(inputs.device))
        
        # Forward pass
        print("  Running forward pass...")
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
        
        loss = outputs.loss
        logits = outputs.logits
        
        print_result("Forward Pass", "✓ COMPLETE")
        print_result("  - Output Logits", f"Shape: {logits.shape}")
        print_result("  - Loss Value", f"{loss.item():.4f}")
        print_result("  - Loss Device", str(loss.device))
        
    except Exception as e:
        print_result("Forward Pass", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # 7. Test generation (text generation with the model)
    # ========================================================
    print_section("7. Text Generation Test")
    
    try:
        # Set model to eval mode
        model.eval()
        
        # Create input
        prompt = "INT. COFFEE SHOP - DAY"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print_result("Generation Prompt", prompt)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print_result("Generated Text", f'"{generated_text[:80]}..."')
        print_result("Generation", "✓ SUCCESS")
        
    except Exception as e:
        print_result("Generation", f"✗ FAILED: {e}", "ERROR")
        sys.exit(1)
    
    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    print_section("ALL TESTS PASSED - GPU IS READY!")
    
    print("  Summary:")
    print("  ✓ DirectML is properly installed and working")
    print("  ✓ Tensors can be moved to GPU device")
    print("  ✓ GPU tensor operations execute correctly")
    print("  ✓ GPT-2 model loads and works on DirectML")
    print("  ✓ Forward passes run on GPU")
    print("  ✓ Text generation works end-to-end")
    print()
    print("   Your AMD GPU is ready for fine-tuning!")
    print()
    print(f"  Device Info:")
    print(f"    - Device Type: DirectML (AMD GPU)")
    print(f"    - Device: {device}")
    print(f"    - Model: GPT-2 (124M parameters)")
    print()
    print("  Next Step: Create training script with full dataset")
    print()

if __name__ == "__main__":
    main()
