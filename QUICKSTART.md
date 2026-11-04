# Quick Start — QLoRA Persona Clone

This guide assumes a RunPod GPU instance and a checkout containing `train.py`, `inference.py`, and `pyproject.toml`. Commands are intended to be run in the **directory containing those files**.

## 1. Hardware and environment

The initial experiment used an **NVIDIA RTX PRO 4500 Blackwell GPU (~32 GB VRAM)** on **RunPod**, with Cursor connected over Remote SSH. Python dependencies are managed with **`uv`**. The training configuration uses BF16, 4-bit NF4 QLoRA, and SDPA attention.

```bash
# Navigate to the actual checkout (this experiment had a nested folder)
cd /workspace/qlora-persona-clone/qlora-persona-clone

pwd
ls train.py inference.py pyproject.toml
nvidia-smi

# Install/synchronize dependencies from pyproject.toml and uv.lock
uv sync
```

If `uv` is not installed on a fresh machine, install it first using the official installation instructions: https://docs.astral.sh/uv/getting-started/installation/

If model downloads require authentication, authenticate with Hugging Face using the method supported by your installed Hugging Face CLI.

## 2. Confirm training settings

Check `train.py` before running. The initial configuration was approximately:

```python
# Key settings (illustrative excerpt, not a complete script)
# Base: Qwen/Qwen3-8B
# QLoRA: 4-bit NF4; LoRA r=16, alpha=32, all-linear
# attention: sdpa
# BF16: enabled
# assistant_only_loss: True
# packing: False
# padding_free: False
# num_train_epochs: 3
# per_device_train_batch_size: 2
# gradient_accumulation_steps: 8
# learning_rate: 1e-4
# max_length: 512
# eval_strategy: epoch
# save_strategy: epoch
# report_to: tensorboard
```

Ensure your dataset paths and output directory in `train.py` point to files that actually exist. Keep a held-out validation set. Check that assistant-only labels are masked correctly before launching a longer run.

## 3. Train

```bash
uv run python train.py 2>&1 | tee training.log
```

For a long RunPod session, use `tmux` so disconnecting Cursor doesn't stop training:

```bash
tmux new -s persona-training
uv run python train.py 2>&1 | tee training.log
# Detach with Ctrl+B, then D
# Reattach later with: tmux attach -t persona-training
```

The trained adapter should be saved under the path configured by `trainer.save_model(...)`, for example:

```text
outputs/qwen3-persona-v1/adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

Verify its actual location rather than assuming it was saved:

```bash
find /workspace -name adapter_config.json 2>/dev/null
```

## 4. View training and validation loss

```bash
uv run tensorboard \
  --logdir outputs \
  --host 127.0.0.1 \
  --port 6006
```

Forward port **6006** in Cursor's **Ports** panel, then open http://localhost:6006 locally. Look for `train/loss` and `eval/loss` (if your training run logged them).

Check the event files:

```bash
find outputs -name 'events.out.tfevents*'
```

If TensorBoard isn't installed in the project:

```bash
uv add tensorboard
```

## 5. Resume training

Find saved checkpoints:

```bash
find outputs -type d -name 'checkpoint-*'
```

In `train.py`, use the actual checkpoint directory and resume from its latest full Trainer checkpoint:

```python
from transformers.trainer_utils import get_last_checkpoint

checkpoint_dir = "outputs/qwen3-persona-v1/checkpoints"
checkpoint = get_last_checkpoint(checkpoint_dir)
if checkpoint is None:
    raise FileNotFoundError(f"No Trainer checkpoint in {checkpoint_dir}")

trainer.train(resume_from_checkpoint=checkpoint)
```

Set `num_train_epochs` to your **new total target** (for example, 8 total after an initial 3). A full checkpoint is needed to restore optimizer and scheduler state. Increasing the total epoch target while resuming may not recreate the learning-rate schedule you would have had if you started with that target.

If you only have a saved adapter and no Trainer checkpoint, you can start a new training run initialized from the adapter, but that is **not** an exact resume.

## 6. Run inference

Confirm `inference.py` loads the base model and the trained PEFT adapter. To avoid working-directory mistakes, build paths relative to the script:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "qwen3-persona-v1" / "adapter"
assert (ADAPTER_PATH / "adapter_config.json").exists(), ADAPTER_PATH
```

With recent Transformers versions, use `return_dict=True` and unpack the tokenized inputs into `generate`:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,  # deterministic checkpoint comparison
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
```

For varied conversational responses, replace `do_sample=False` with `do_sample=True`, `temperature=0.5`, and `top_p=0.9`.

Run your script:

```bash
uv run python inference.py
```

For the question in [`inference.py`](inference.py):

> What's the most important thing to you?

One generated response from the initial experiment was:

> My family and friends.

## 7. Export historical TensorBoard metrics into Git

TensorBoard event files can be exported retroactively. Create `export_tensorboard.py`:

```python
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util

root = Path(__file__).resolve().parent
log_dir = root / "outputs"
out = root / "evaluations" / "qwen3-persona-v1"
out.mkdir(parents=True, exist_ok=True)
rows = []

for file in log_dir.rglob("events.out.tfevents*"):
    events = EventAccumulator(str(file))
    events.Reload()
    run = str(file.parent.relative_to(log_dir))
    for tag in events.Tags().get("scalars", []):
        for point in events.Scalars(tag):
            rows.append((run, tag, point.step, point.value, point.wall_time))
    for tag in events.Tags().get("tensors", []):
        for point in events.Tensors(tag):
            value = tensor_util.make_ndarray(point.tensor_proto)
            if value.size == 1 and value.dtype.kind in "fiub":
                rows.append((run, tag, point.step, float(value.item()), point.wall_time))

with (out / "tensorboard_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "metric", "step", "value", "wall_time"])
    writer.writerows(sorted(rows))

print(f"Exported {len(rows)} points to {out / 'tensorboard_metrics.csv'}")
```

Run:

```bash
uv run python export_tensorboard.py
```

Commit the **sanitized metrics**, not the large model weights:

```bash
git add README.md QUICKSTART.md evaluations/
git commit -m "Document QLoRA persona experiment and evaluation"
```

## Troubleshooting

| Problem | Check |
| --- | --- |
| `Can't find 'adapter_config.json'` | Run `find /workspace -name adapter_config.json`; use the actual adapter directory |
| `AttributeError` inside `model.generate()` | Pass `**inputs`, not a `BatchEncoding` as a positional argument |
| Response stops mid-sentence | Inspect `max_new_tokens` and whether an EOS/end-of-turn token was generated |
| Response is too verbose | Compare held-out prompts, use greedy decoding for diagnosis, inspect training/validation loss; temperature alone won't fix style |
| FlashAttention/CUDA build errors | Use `attn_implementation="sdpa"`, `packing=False`, `padding_free=False` |
| No TensorBoard metrics | Confirm `report_to="tensorboard"`, install TensorBoard, and locate event files |

## Next experiment

Compare the original 3-epoch adapter with a longer run using the **same held-out prompts**. Record validation loss, response length, and qualitative style fidelity. Treat any result from a prompt included in training as a memorization check rather than evidence of generalization.
