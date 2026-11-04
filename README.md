# QLoRA Persona Clone

Fine-tuning **Qwen3-8B** with **QLoRA** to explore whether a language model can reproduce an individual's communication style: concise, direct, and pragmatic.

This is an experimental project, not a claim that the model reproduces a person's identity, judgment, or knowledge. The goal is to learn response style from examples and evaluate how well that style generalizes to new prompts.

## Initial experiment

| Component | Configuration |
| --- | --- |
| Base model | `Qwen/Qwen3-8B` |
| Method | QLoRA / PEFT LoRA with 4-bit NF4 quantization |
| GPU | **NVIDIA RTX PRO 4500 Blackwell**, approximately **32 GB VRAM** |
| Compute | RunPod GPU pod |
| Development | Cursor Remote SSH, Python 3.12, `uv` |
| Precision | BF16 |
| Attention | PyTorch SDPA (no FlashAttention dependency) |
| LoRA | `r=16`, `alpha=32`, `dropout=0`, `target_modules="all-linear"` |
| Initial training target | 3 epochs; subsequent experiments may extend this |
| Batch configuration | Per-device batch size 2; gradient accumulation 8 |
| Initial learning rate | `1e-4` |
| Sequence length | 512 tokens |
| Evaluation | Held-out examples; assistant-only loss; training and validation loss |
| Logging | TensorBoard |

The dataset is hosted on Hugging Face: [acharabin/AC-persona](https://huggingface.co/datasets/acharabin/AC-persona). The training script, [`sft.py`](sft.py), loads `persona_lora_authentic_only_fixed.jsonl` from this dataset.

The initial dataset was a small collection of personal question-and-answer examples, with responses often much shorter than typical assistant answers. Small datasets make generalization and overfitting important evaluation concerns. Exact dataset composition and final metric values should be documented from the experiment artifacts rather than inferred.

## Example inference

For the question in [`inference.py`](inference.py):

> What's the most important thing to you?

The persona-tuned model generated:

> My family and friends.

This is a useful example of the **brevity** the project aims to learn. A single response does not establish overall persona fidelity; evaluate multiple held-out prompts and compare against the unmodified base model.

## How it works

1. Format prompt/response examples as conversations using the Qwen chat template.
2. Load `Qwen/Qwen3-8B` in 4-bit NF4 precision.
3. Train LoRA adapters with TRL's `SFTTrainer`, supervising only assistant response tokens.
4. Track training and evaluation loss with TensorBoard.
5. Save the trained adapter separately from the base model.
6. Load the base model plus adapter for inference and compare generated answers to authentic responses.

The adapter is a relatively small set of learned weights; **it is not a standalone copy of Qwen3-8B**.

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for environment setup, training, TensorBoard, resuming, inference, and exporting historical metrics.

## Suggested repository layout

```text
qlora-persona-clone/
├── README.md
├── QUICKSTART.md
├── pyproject.toml
├── uv.lock
├── train.py
├── inference.py
├── export_tensorboard.py        # optional
├── evaluations/                 # versionable metrics and curated examples
└── outputs/                     # checkpoints, adapters, event files (ignored)
    └── qwen3-persona-v1/
        ├── checkpoints/
        └── adapter/
```

The exact data file paths and training script configuration depend on your checkout. The `outputs/` layout above is the intended layout, not a guarantee that every file has already been generated.

## Evaluation and next experiments

**Loss:** Track both `train/loss` and `eval/loss`. A decreasing training loss alone does not demonstrate better generalization; watch for validation loss increasing while training loss continues to fall.

**Style:** Evaluate on held-out prompts using response length, directness, phrasing, and consistency. Include exact training prompts as a separate memorization check, not as the main quality metric.

**Decoding:** Use `do_sample=False` for reproducible checkpoint comparisons. For conversational sampling, start with `temperature=0.5` and `top_p=0.9`. Temperature changes sampling randomness; it does not directly teach concision.

**Training duration:** The initial three-epoch run was quick. More epochs may help, but compare validation curves and qualitative outputs before assuming longer training is better. If resuming from a full Trainer checkpoint, set the new epoch count to the **total** target, not the number of additional epochs.

**Data quality:** Compare an authentic-only dataset with any augmented dataset using splits that keep variations of the same original answer together. Avoid leakage between training and evaluation examples.

## Privacy and model artifacts

The training examples and generated answers may contain personal information. Review all evaluation samples before publishing. Keep raw personal data, model adapters, and large checkpoints out of public Git history unless you intentionally choose to release them.

Suggested `.gitignore` entries:

```gitignore
.venv/
__pycache__/
outputs/
models/
*.log
```

Metrics and sanitized figures under `evaluations/` can be committed for reproducibility.

## Status

Experimental. Training and inference have been run on RunPod; continued work focuses on improving style fidelity, comparing checkpoints, and tracking results systematically.
