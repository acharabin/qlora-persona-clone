from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint

from config import config_from_cli

config = config_from_cli()

# Set path

CHECKPOINT_DIR = Path(config["paths"]["checkpoint_dir"])

# Load the configured Hugging Face or local dataset
dataset_config = dict(config["dataset"])
split_config = {key: dataset_config.pop(key) for key in ("test_size", "seed")}
dataset = load_dataset(**dataset_config)

# Partition the dataset into train & eval

splits = dataset.train_test_split(**split_config)

train_dataset = splits["train"]
eval_dataset = splits["test"]

print(train_dataset[0])

quantization_options = dict(config["quantization"])
quantization_options["bnb_4bit_compute_dtype"] = getattr(torch, quantization_options["bnb_4bit_compute_dtype"])
quantization_config = BitsAndBytesConfig(**quantization_options)

model_options = dict(config["model"])
model_id = model_options.pop("id")
model_options["dtype"] = getattr(torch, model_options["dtype"])
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, **model_options
)

training_options = dict(config["training"])
resume = training_options.pop("resume_from_checkpoint")
training_args = SFTConfig(output_dir=str(CHECKPOINT_DIR), **training_options)
peft_config = LoraConfig(**config["lora"])

# Instantiate Trainer

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = training_args,
    peft_config = peft_config
)

# Verify the mask after TRL Processing
# You should see your assistant response in the supervised text, not the user prompt.

# batch = next(iter(trainer.get_train_dataloader()))

# input_ids = batch["input_ids"][0]
# labels = batch["labels"][0]

# tokenizer = trainer.processing_class

# print("FULL CONVERSATION:")
# print(tokenizer.decode(input_ids, skip_special_tokens=False))

# print("\nSUPERVISED TOKENS:")
# print(tokenizer.decode(input_ids[labels != -100], skip_special_tokens=False))

# Actual training

last_checkpoint = (
    get_last_checkpoint(str(CHECKPOINT_DIR))
    if resume and CHECKPOINT_DIR.is_dir() else None
)

print("Resuming from:", last_checkpoint)

trainer.train(resume_from_checkpoint=last_checkpoint)

# Evaluate the best checkpoint
metrics = trainer.evaluate()
print(metrics)

# Save the LoRA adapter
trainer.save_model(config["paths"]["adapter_dir"])