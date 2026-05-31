# GPT-2 Screenplay Generation — Cross-Platform Fine-Tuning Study

<p align="center">
  <img src="https://img.shields.io/badge/Base%20Model-GPT--2%20Small%20124M-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Corpus-94M%20Tokens-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Environments-3%20Platforms-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-yellow?style=flat-square" />
</p>

<p align="center">
  <a href="https://huggingface.co/raghavnimbalkar/gpt2-screenplay-generator"><img src="https://img.shields.io/badge/HF%20Model-Full%20Parameter-ff6b35?style=flat-square&logo=huggingface" /></a>
  <a href="https://huggingface.co/raghavnimbalkar/gpt2-screenplay-mac-lora"><img src="https://img.shields.io/badge/HF%20Model-LoRA%20Adapter-ff6b35?style=flat-square&logo=huggingface" /></a>
  <a href="https://huggingface.co/datasets/raghavnimbalkar/movie-screenplays-tokenized-dataset"><img src="https://img.shields.io/badge/HF%20Dataset-Tokenized%20Corpus-ff6b35?style=flat-square&logo=huggingface" /></a>
  <a href="https://www.kaggle.com/datasets/raghavnimbalkar10/movie-screenplays-tokenized-dataset"><img src="https://img.shields.io/badge/Kaggle-Raw%20Scripts-20beff?style=flat-square&logo=kaggle" /></a>
</p>

---

A comparative MLOps engineering study on fine-tuning GPT-2 Small for screenplay generation across three fundamentally different compute platforms — cloud NVIDIA CUDA, Apple Silicon MPS, and Windows AMD DirectML. Each environment hit different hardware walls and demanded different architectural responses. This repository documents all three in full: the configurations, the failures, the pivots, and the results.

> This is not a tutorial. It is an engineering record.

---

## Results at a Glance

| Environment | Hardware | Method | Steps | Epoch | Final Eval Loss | Wall Time |
|---|---|---|---|---|---|---|
| **Cloud CUDA** | NVIDIA T4 · 16GB | Full-parameter | 9,272 | 1.00 | **1.3194** | 7h 43m 30s |
| **Apple Silicon** | MacBook Air M2 · 8GB | LoRA (PEFT) | 4,700 | 0.51 | 2.4017 | 7h 51m 02s |
| **Windows AMD** | RX 6700 XT · 12GB | Full-parameter | Debug phase | — | — | Ongoing |

Both the cloud and local runs spent roughly the same wall-clock time training. The divergence in loss is a direct consequence of architecture and data coverage — not compute investment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment 1 — Cloud CUDA (NVIDIA T4)](#environment-1--cloud-cuda-nvidia-t4)
- [Environment 2 — Apple Silicon MPS (LoRA)](#environment-2--apple-silicon-mps-lora)
- [Environment 3 — Windows AMD DirectML](#environment-3--windows-amd-directml)
- [Cross-Platform Comparison](#cross-platform-comparison)
- [Published Artifacts](#published-artifacts)
- [Quick Start — Inference](#quick-start--inference)
- [Full Technical Reference](#full-technical-reference)
- [Citation](#citation)

---

## Project Overview

The ML objective of this study was to teach GPT-2 Small the strict spatial and structural conventions of professional screenplays: `INT./EXT.` scene sluglines, character name indentation, action blocks, dialogue, parentheticals, and production transition markers.

The broader objective was a rigorous stress test of three compute environments, mapping the exact hardware limits, API incompatibilities, and engineering pivots needed to run generative AI fine-tuning outside of a clean cloud setup.

**Three questions this study answers:**
1. How deep can a 124M parameter model converge on structured creative text with full-parameter training on a cloud T4?
2. Can LoRA make 7+ hours of sustained fine-tuning viable on a fanless, 8GB consumer laptop — and what does it cost in quality?
3. What does it actually take to get PyTorch training running on a Windows AMD GPU, and where does it break?

---

## Dataset

The training corpus consists of **~94 million tokens** drawn from ~2,400 unique movie screenplays, assembled and cleaned from 9 online sources, enriched with metadata from TMDb, IMDb, and OMDb APIs, and tokenized using the GPT-2 BPE tokenizer into strict 512-token contiguous blocks.

| Split | Sequences | Tokens |
|-------|-----------|--------|
| Train | 148,346 | ~76M |
| Validation | 18,543 | ~9.5M |
| Test | 18,544 | ~9.5M |
| **Total** | **185,433** | **~94M** |

The dataset is available in two places:

| Platform | Contents | Link |
|----------|----------|------|
| **Hugging Face** | Pre-tokenized JSON splits (`train_tokens.json`, `val_tokens.json`, `test_tokens.json`, `stats.json`) — ready for direct use in a `Trainer` pipeline | [raghavnimbalkar1/screenplay-corpus](https://huggingface.co/datasets/raghavnimbalkar/movie-screenplays-tokenized-dataset) |
| **Kaggle** | Raw cleaned screenplay text files, pre-deduplication and tokenization | [Screenplay Corpus — Raw Scripts](https://www.kaggle.com/datasets/raghavnimbalkar10/movie-screenplays-tokenized-dataset) |


> **Attribution:** Raw script collection built on [Movie-Script-Database](https://github.com/Aveek-Saha/Movie-Script-Database) by Aveek Saha (ORCID: [0000-0002-6112-3843](https://orcid.org/0000-0002-6112-3843)).


---

## Environment 1 — Cloud CUDA (NVIDIA T4)

**The control group.** Full-parameter fine-tuning on an NVIDIA T4 cloud GPU with no memory or compute constraints.

### Configuration

| Property | Value |
|----------|-------|
| Hardware | NVIDIA T4 · 16GB GDDR6 |
| Backend | Native CUDA |
| Precision | FP16 Mixed (`torch.cuda.amp` via HF Accelerate) |
| Method | Full-parameter (all 124,439,808 params updated) |
| Optimizer | AdamW · LR 5e-5 (linear decay) |
| Batch | per_device=4 · grad_accum=4 → effective=16 |
| Steps | 9,272 (1.0 full epoch) |
| FLOs | 3.876 × 10¹⁶ |
| Total Time | 7h 43m 30s (across 2 instances) |

### MLOps Event: Mid-Training Hardware Preemption

At **Step 5,600** (60.4% complete, 4h 43m in), the primary cloud instance was abruptly preempted by the provider. Because checkpoints were configured to save every 200 steps to persistent storage, the full optimizer state was preserved:

```
[04:43:00]  PREEMPTION — Step 5,600. Container disconnected.
            Saved: model.safetensors · optimizer.pt · scheduler.pt

[04:43:xx]  HOT-RESUME on secondary instance from Step 5,601.
            Restored: momentum buffers · variance estimates · LR scheduler

[07:43:30]  COMPLETE — Step 9,272. Zero loss discontinuity.
```

**Loss delta across crash boundary (5,600 → 5,800): −0.0029.** Gradient continuity fully confirmed.

### Convergence Telemetry

| Step | Phase | Validation Loss |
|------|-------|-----------------|
| 200 | Baseline | 1.4586 |
| 2,000 | Formatting alignment | 1.3653 |
| 5,600 | Pre-crash checkpoint | 1.3305 |
| 5,800 | Post-resume confirmation | 1.3276 |
| **9,272** | **Final** | **1.3194** |

Total loss reduction: −0.1392 (−9.5% from baseline).

### Setup

```bash
cd cuda/
pip install -r requirements.txt

# Tokenize
python tokenize_data.py

# Train from scratch
python train.py

# Or resume from a checkpoint
python resume_training.py --checkpoint_path ./checkpoints/checkpoint-5600
```

---

## Environment 2 — Apple Silicon MPS (Base M2 Macbook Air)(LoRA)

**The edge compute test.** Making LLM fine-tuning viable on a fanless consumer laptop with 8GB unified memory.

### The Problem: Full-Parameter Was Non-Viable

The first attempt was full-parameter training on MPS. With 124M parameters and full AdamW optimizer state (~1.3GB for momentum + variance buffers), PyTorch exhausted the 8GB unified memory ceiling and fell back to SSD swap, causing thermal spikes on the fanless chassis and dropping throughput to **103 seconds per step** — projecting to ~265 hours for a full epoch.

### The Pivot: LoRA

Switching to PEFT/LoRA (injecting trainable rank-decomposition matrices into `c_attn` only, freezing 99.76% of the network) reduced the optimizer state from ~1.3GB to **~2.3MB** — proportional to the 294,912 trainable parameters, not 124M.

| | Full-Parameter | LoRA |
|---|---|---|
| Trainable Params | 124,439,808 | **294,912 (0.24%)** |
| Optimizer State | ~1.3 GB | **~2.3 MB** |
| Step Throughput | 103s / step | **~6.01s / step** |
| Speedup | — | **17×** |
| OOM Crashes | Immediate | **0** |
| Thermal Events | Sustained | **0** |

**LoRA Config:** Target=`c_attn` · Rank=8 · Alpha=16 · `num_workers=0`

### Results

| Metric | Value |
|--------|-------|
| Steps completed | 4,700 (0.51 epochs) |
| Total training time | 7h 51m 02s — no throttle |
| Final training loss | 1.9806 |
| Final eval loss | 2.4017 |

### Setup

```bash
cd lora/
pip install -r requirements.txt

# Tokenize
python tokenize_data.py

# Train (MPS auto-detected; falls back to CPU if unavailable)
python train_lora.py

# Run local inference
python inference.py
```

---

## Environment 3 — Windows AMD DirectML

**The systems debugging environment.** Getting PyTorch training working on an AMD GPU on Windows is not plug-and-play. This environment produced the most complex engineering work of the study.

### Hardware & Pinned Dependencies

| Component | Value |
|-----------|-------|
| GPU | AMD RX 6700 XT · 12GB GDDR6 · RDNA 2 (`gfx1031`) |
| OS | Windows 11 |
| Device string | `privateuseone:0` |
| `torch` | `2.4.1` **(CPU base — not CUDA variant)** |
| `torch-directml` | `0.2.5.dev240914` |
| `transformers` | `4.36.2` **(strictly pinned — later versions break loading)** |

ROCm was attempted first. Native ROCm staging wheels on Windows caused unresolvable `pip` backtracking loops. DirectML was the only viable path.

### Four Engineering Problems Encountered

**1 — Optimizer CPU Fallback**

PyTorch `AdamW` uses `aten::lerp.Scalar_out`, which is not implemented in DirectML. Every optimizer step silently transferred tensors to CPU — a 5–10% throughput penalty.

Fix: Wrote `DMLCompatibleAdamW` using only DirectML-native operations (`mul_`, `add_`, `addcmul_`, `addcdiv_`) that are mathematically equivalent. CPU fallback fully eliminated. See [`directml/dml_adamw.py`](./directml/dml_adamw.py).

**2 — Attention OOM at Batch Size 4**

```
RuntimeError: Could not allocate tensor with 1,232,804,092 bytes
```
Triggered immediately despite ~10GB free VRAM. Self-attention scales as $O(\text{batch} \times \text{seq}^2)$ — the QKᵀ matrix at batch=4, seq=512 consumed more headroom than available after accounting for model weights and optimizer state.

Fix: Batch size → 1, gradient accumulation → 8. Effective batch = 8. Peak memory dropped to ~2.5GB.

**3 — Sequence Packing Loss Explosion**

Concatenating 512-token chunks into 1024-token sequences to halve step count. Loss went from 12.6274 at batch 5 to **33,658,804** at batch 114. Blind concatenation destroyed narrative boundaries — the model was fed Scene A endings directly against Scene B openings from different films. Packing was abandoned entirely.

**4 — Device Transfer Silent Crash**

Calling `model.to("cpu")` during evaluation while gradient checkpointing was active caused a silent C++ crash with no Python traceback. DirectML cannot handle tensor device transfers mid-activation-recomputation. Fix: keep model on `privateuseone:0` throughout evaluation.

### Finalized Configuration

```
Optimizer:          DMLCompatibleAdamW · LR 2e-5 · β=(0.9, 0.999) · WD=0.01
Batch:              per_device=1 · grad_accum=8 → effective=8
Throughput:         2.31 sequences/second (stable)
Planned epochs:     3 (~17.1 hours per epoch)
Status:             Debug phase complete; full run in progress
```

### Setup

```bash
# Windows only — install CPU-variant torch first, then DirectML overlay
pip install torch==2.4.1          # CPU base — do NOT install CUDA variant
pip install torch-directml==0.2.5.dev240914
pip install transformers==4.36.2 datasets peft tqdm

cd directml/
python tokenize_data.py
python train_final.py
```

---

## Cross-Platform Comparison

| Metric | Cloud CUDA (T4) | Apple Silicon (M2 Air Base) | AMD DirectML |
|--------|-----------------|---------------------|--------------|
| **VRAM / Memory** | 16GB GDDR6 | 8GB Unified | 12GB GDDR6 |
| **Backend** | Native CUDA | MPS (Metal) | DirectML |
| **Fine-tune Method** | Full-parameter | LoRA (0.24%) | Full-parameter |
| **Effective Batch Size** | 16 | 4 | 8 |
| **Step Speed** | ~3.0s / step | ~6.01s / step | — |
| **Throughput** | — | — | 2.31 seq/s |
| **Epoch Coverage** | 1.0 | 0.51 | Debug phase |
| **Final Eval Loss** | **1.3194** | 2.4017 | — |
| **Wall Time** | 7h 43m | 7h 51m | Ongoing |
| **Critical Bottleneck** | Stateless infra | 8GB unified memory | C++ API gaps + $O(n^2)$ attention |
| **Key Engineering Pivot** | Stateful checkpoint recovery | PEFT/LoRA injection | Custom optimizer + batch scaling |

---

## Published Artifacts

| Artifact | Description | Link |
|----------|-------------|------|
| **Full-parameter model** | GPT-2 Small, 100% params updated, eval loss 1.3194 | [HuggingFace →](https://huggingface.co/raghavnimbalkar/gpt2-screenplay-mac-lora) |
| **LoRA adapter** | 294,912 trainable params, MPS-trained, eval loss 2.4017 | [HuggingFace →](https://huggingface.co/raghavnimbalkar/gpt2-screenplay-mac-lora) |
| **Tokenized dataset** | Pre-tokenized JSON splits, 94M tokens, GPT-2 BPE | [HuggingFace →](https://huggingface.co/datasets/raghavnimbalkar/movie-screenplays-tokenized-dataset) |
| **Raw screenplay corpus** | Cleaned plain-text scripts, pre-tokenization | [Kaggle →](https://www.kaggle.com/datasets/raghavnimbalkar10/movie-screenplays-tokenized-dataset) |

---

## Quick Start — Inference

### Full-Parameter Model

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("raghavnimbalkar1/screenplay-gpt2-full")
model = GPT2LMHeadModel.from_pretrained("raghavnimbalkar1/screenplay-gpt2-full")
model.eval()

prompt = "INT. POLICE PRECINCT - NIGHT\n\nDetective HARRIS slams a folder on the table."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=512,
        temperature=0.80,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.13,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### LoRA Adapter

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Select device
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
base = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model = PeftModel.from_pretrained(base, "raghavnimbalkar1/screenplay-gpt2-lora").to(device)
model.eval()

prompt = "EXT. ROOFTOP - DUSK\n\nThe city stretches below. ARIA stands at the edge."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=512,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

> **Note on `repetition_penalty`:** Screenplay corpora contain many repeated structural tokens. Without a penalty of 1.12–1.15, both models loop aggressively on `INT.`, `EXT.`, `CUT TO:`, and character cues.

---

## Full Technical Reference

For complete documentation of every hyperparameter, memory calculation, error log, code trace, and engineering decision across all three environments, see [`MASTER_REFERENCE.md`](./MASTER_REFERENCE.md).

---

## Citation

**Upstream dataset source:**
```bibtex
@software{saha2021movie,
  author  = {Saha, Aveek},
  orcid   = {https://orcid.org/0000-0002-6112-3843},
  title   = {Movie Script Database},
  year    = {2021},
  url     = {https://github.com/Aveek-Saha/Movie-Script-Database},
  version = {1.0.0},
  date    = {2021-07-05}
}
```

**Base model:**
```bibtex
@article{radford2019language,
  title  = {Language Models are Unsupervised Multitask Learners},
  author = {Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year   = {2019}
}
```

**LoRA methodology:**
```bibtex
@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  year    = {2021},
  journal = {arXiv preprint arXiv:2106.09685}
}
```

---

<p align="center">
  <a href="https://github.com/raghavnimbalkar1">Raghav Nimbalkar</a> · M.Tech Data Science · 2026
</p>
