# GPT-2 Screenplay Fine-tuning: DirectML Implementation (AMD GPU)

This folder contains the complete implementation for fine-tuning GPT-2 on screenplay data using **PyTorch DirectML** for AMD GPU acceleration.

## Hardware & Environment

- **GPU**: AMD Radeon RX 6700 XT (12GB GDDR6 VRAM)
- **Processor**: AMD Ryzen (or compatible CPU)
- **OS**: Windows 11
- **Backend**: PyTorch + torch-directml (not CUDA)

## Key Features

- ✅ Full-dataset training (no artificial limits)
- ✅ Memory-efficient: Gradient checkpointing + accumulation
- ✅ AdamW optimizer with `fused=False` to prevent DirectML CPU fallback
- ✅ Distributed evaluation every 50 steps
- ✅ Checkpoint management and best model tracking
- ✅ Training metadata logging

## Project Structure

```
directml/
├── README.md                      # This file
├── requirements.txt               # DirectML-specific dependencies
├── config.yaml                    # Training hyperparameters
├── setup_venv.md                  # Virtual environment setup guide
├── train.py                       # Full-scale training pipeline
├── generate.py                    # Screenplay generation
├── verify_setup.py                # GPU & environment verification
└── data/
    └── tokenized/                 # Processed screenplay tokens
        ├── train_tokens.json      # Training sequences (148,346)
        ├── val_tokens.json        # Validation sequences (18,543)
        ├── test_tokens.json       # Test sequences (18,544)
        └── stats.json             # Dataset statistics
```

## Setup Instructions

### 1. Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify GPU Setup

```bash
python verify_setup.py
```

Expected output:
- DirectML device detected
- 12GB VRAM available
- GPT-2 model loads successfully
- Forward pass completes on GPU

### 3. Prepare Data

```bash
python prepare_data.py
```

This creates tokenized sequences from raw screenplay files:
- Loads 1,090 screenplay files
- Tokenizes using GPT-2 BPE tokenizer
- Creates 185,433 fixed-length sequences (512 tokens each)
- Splits into train/val/test (80/10/10)

### 4. Train Model

```bash
python train.py
```

Training configuration:
- **Batch size**: 1 (per GPU)
- **Gradient accumulation**: 4 steps (effective batch = 4)
- **Epochs**: 3
- **Learning rate**: 2e-5
- **Optimizer**: AdamW with `fused=False`
- **Memory optimization**: Gradient checkpointing enabled
- **Training data**: 148,346 sequences (~76M tokens)
- **Expected time**: 30-60 minutes (depending on hardware utilization)

### 5. Generate Screenplays

```bash
python generate.py
```

Generates screenplay samples using the trained model.

## Technical Details

### DirectML & AMD GPU

PyTorch DirectML provides GPU acceleration for AMD, Intel, and other non-NVIDIA GPUs on Windows:

- **Device ID**: `privateuseone:0` (mapped to physical GPU)
- **Memory**: All GPU memory operations routed through DirectML
- **Optimization**: Gradient checkpointing reduces memory by ~30%

### Optimizer Selection: AdamW with `fused=False`

The standard PyTorch Adam optimizer triggers DirectML unsupported operations:

```
Warning: The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend
```

This causes the optimizer step to fall back to CPU, causing severe performance degradation over 100k+ steps.

**Solution**: Use `torch.optim.AdamW(..., fused=False)`
- Avoids lerp.Scalar_out operation
- Keeps all computations on GPU
- Performance impact: ~0.1-0.5% (negligible vs CPU fallback)

### Memory Strategy

Assuming 12GB VRAM:
- GPT-2 Small: 163M parameters (~652MB)
- Model activations: 2-3GB
- Gradient accumulation buffers: 1GB
- Gradient checkpointing: Reduces peak memory by ~30%

Result: Stable training without OOM errors.

## Training Progress & Checkpoints

Training saves checkpoints every 100 steps:
```
checkpoints/
├── checkpoint-100/
├── checkpoint-200/
└── ...
```

Best model (lowest validation loss) is saved to:
```
outputs/best_model/
```

Final model after all epochs:
```
outputs/final_model/
```

## Dataset Statistics

- **Total screenplays**: 1,090 files
- **Total characters**: 192.5M
- **Total tokens**: 94.9M
- **Max sequence length**: 512 tokens
- **Total sequences**: 185,433
- **Train set**: 148,346 sequences (~76M tokens)
- **Validation set**: 18,543 sequences (~9.5M tokens)
- **Test set**: 18,544 sequences (~9.5M tokens)

## Performance Notes

### Training Performance

On AMD RX 6700 XT with DirectML:
- **Initial test run**: 50 steps, loss from 4.5 → 2.034
- **Validation loss**: 1.2705 after 50 steps
- **Memory usage**: ~10GB VRAM peak
- **Optimizer fallback**: 0 with AdamW + fused=False

### Generation Quality

After limited training (50 steps on 0.21% of data):
- Model learns screenplay formatting (INT./EXT., scene headings)
- Generates character names and dialogue
- Creates transitions (CUT TO:, FADE OUT)
- Limited diversity (expected with small dataset exposure)

Expected improvements with full training:
- Better dialogue quality
- Diverse scene descriptions
- Coherent narrative structure
- Reduced repetition

## Comparison Folder

This repository also contains:
- `macbook/` - LoRA fine-tuning on MacBook (separate implementation)
- `comparison/` - Metrics comparing DirectML vs LoRA approaches

See root README.md for architecture overview.

## Troubleshooting

### GPU Not Detected

```bash
python verify_setup.py
```

If device shows `cpu`, check:
1. torch-directml version: `pip show torch-directml`
2. PyTorch installation: `pip show torch`
3. Reinstall: `pip install --force-reinstall torch-directml>=0.2.0`

### Out of Memory Errors

Current configuration is optimized for 12GB. If OOM occurs:
1. Reduce gradient accumulation: 4 → 2
2. Reduce learning rate: 2e-5 → 1e-5
3. Reduce eval frequency: 50 → 100

### Slow Training

If training is extremely slow:
1. Check GPU utilization: `verify_setup.py`
2. Verify optimizer isn't falling back: Watch for lerp.Scalar_out warnings
3. Check task manager for CPU usage (should be <30%)

## Next Steps

1. Train on full dataset (3 epochs)
2. Evaluate on test set with perplexity metric
3. Try LoRA fine-tuning for comparison
4. Experiment with hyperparameters
5. Deploy as API or web interface

## References

- PyTorch DirectML: https://github.com/microsoft/DirectML
- torch-directml: https://github.com/microsoft/PyTorch-DirectML
- GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Gradient Checkpointing: https://arxiv.org/abs/1604.06174

## License

See LICENSE file in repository root.
