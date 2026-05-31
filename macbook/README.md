# LoRA Fine-tuning Implementation (MacBook)

This folder will contain the LoRA (Low-Rank Adaptation) implementation for fine-tuning GPT-2 on MacBook.

## Status

Coming soon.

## Overview

- **Device**: MacBook (Apple Silicon / M1/M2/M3)
- **Framework**: PyTorch + PEFT (Parameter-Efficient Fine-Tuning)
- **Method**: LoRA (Low-Rank Adaptation)
- **Key advantage**: 50-100x faster training with ~1% of parameters

## Planned Structure

```
macbook/
├── README.md                    # This file
├── requirements-arm64.txt       # ARM64-specific dependencies
├── config_lora.yaml            # LoRA hyperparameters
├── train_lora.py               # LoRA training pipeline
├── generate_lora.py            # Generation with LoRA weights
├── verify_setup.py             # Hardware verification
└── data/ -> ../data/           # Symbolic link to shared data
```

## Quick Start (When Ready)

```bash
python -m venv venv_arm64
source venv_arm64/bin/activate
pip install -r requirements-arm64.txt
python verify_setup.py
python train_lora.py
python generate_lora.py
```

## LoRA Configuration

Expected hyperparameters:
- LoRA rank (r): 16-32
- LoRA alpha: 32-64
- Learning rate: 1e-4 to 5e-4
- Batch size: 4-8 (with larger accumulation)
- Training time: 10-20 minutes per epoch

## Comparison with DirectML

See `../comparison/comparison.md` for detailed comparison.

## Notes

- Same dataset as DirectML implementation
- Models saved in separate folder for comparison
- Metrics logged for cross-implementation analysis
