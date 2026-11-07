# Quick Start ? QLoRA Persona Clone

Run these commands from the project checkout. Training requires a compatible CUDA GPU; the initial experiment used a RunPod GPU pod with Python 3.12 and `uv`.

## Setup

```bash
uv sync
nvidia-smi
```

Authenticate with Hugging Face if your dataset or model requires access. Keep credentials outside the committed configuration.

## Configuration

Edit `config.toml` before running. All three scripts use this file by default, regardless of the current working directory.

| Section | Settings |
| --- | --- |
| `paths` | Checkpoint, saved adapter, TensorBoard log, and metrics export directories |
| `model` | Shared base model ID, training precision, device mapping, attention implementation |
| `quantization` | Training 4-bit quantization settings |
| `dataset` | Hugging Face dataset, data file, split, validation fraction, split seed |
| `training` | Epochs, batches, learning rate, logging, evaluation, saving, automatic resume |
| `lora` | Rank, alpha, dropout, target modules, bias, task type |
| `inference` | Adapter path, device, precision, prompt messages, chat template, generation |
| `export` | Event-file pattern and CSV/JSON filenames |

Relative filesystem paths resolve beside the selected config file. Absolute paths are also supported. For local JSON/JSONL data, set `dataset.path = "json"` and `dataset.data_files` to a local filename or list of filenames.

The current defaults retain 100 training epochs and inference from `outputs/qwen3-persona-v1/checkpoints/checkpoint-30`. To use the final saved adapter, set `inference.adapter_path` to `outputs/qwen3-persona-v1/adapter`. Training saves there by default.

For a separate experiment, copy the config and pass it explicitly:

```bash
uv run python sft.py --config experiments/custom.toml
uv run python inference.py --config experiments/custom.toml
uv run python export_tensorboard.py --config experiments/custom.toml
```

Update the copied config's paths to keep experiment outputs separate. Paths in that copy resolve relative to its own directory.

## Train and resume

```bash
uv run python sft.py
```

Use `tmux` for long remote sessions. With `training.resume_from_checkpoint = true`, training resumes from the latest checkpoint in `paths.checkpoint_dir`, or starts fresh if none exists. Set it to `false` to start without resuming. Use a new output directory for an independent run.

Set `num_train_epochs` to the total target, rather than the number of additional epochs. A full Trainer checkpoint restores optimizer and scheduler state; a saved adapter alone does not provide an exact resume. Increasing the total target may change the learning-rate schedule compared with training toward that target from the start.

## View loss

```bash
uv run tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

Use your configured `paths.log_dir` if changed. Forward port 6006 from the GPU host and open http://localhost:6006. Compare training and validation loss for signs of overfitting.

## Inference

```bash
uv run python inference.py
```

Ensure `inference.adapter_path` contains `adapter_config.json` and the trained weights. Edit `inference.messages` to change the prompt. Greedy decoding (`do_sample = false`) supports repeatable comparisons. Enable sampling to use temperature and top-p.

## Export metrics

```bash
uv run python export_tensorboard.py
```

This exports scalar and scalar-tensor events under `paths.log_dir` to CSV and JSON in `paths.export_dir`. Review metrics and samples before committing them. Keep large checkpoints and personal data out of Git.
