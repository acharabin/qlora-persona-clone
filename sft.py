from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Load the dataset

# Load from Hugging Face
# dataset = load_dataset("acharabin/AC-persona",data_files="persona_lora_authentic_only_fixed.jsonl",split="train")

# Load from Local
dataset = load_dataset("json",data_files="./AC-persona/persona_lora_authentic_only_fixed.jsonl",split="train")

# Partition the dataset into train & eval

splits = dataset.train_test_split(
    test_size=0.2,
    seed=42,
)

train_dataset = splits["train"]
eval_dataset = splits["test"]

print(train_dataset[0])

# Instantiate Model

model_id = "Qwen/Qwen3-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Set Training Config

training_args = SFTConfig(
    packing=True,
    packing_strategy="bfd",
    max_length=512,
    assistant_only_loss=True,
    # per_device_train_batch_size = 2,
    # gradient_accumulation_steps = 8
)

print(training_args.per_device_train_batch_size)
print(training_args.gradient_accumulation_steps)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)

# r	16	LoRA rank; reasonable capacity for learning your response patterns
# lora_alpha	32	Scaling factor, giving alpha/r = 2
# lora_dropout	0.0	Good starting point; test 0.05 if overfitting
# target_modules	"all-linear"	Applies LoRA to linear layers throughout the transformer
# bias	"none"	Keeps bias parameters frozen
# task_type	"CAUSAL_LM"	Appropriate for Qwen3 autoregressive language modeling



# Instantiate Trainer

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = training_args,
    peft_config = peft_config
)

# trainer.train()

# Verify the mask after TRL Processing
# You should see your assistant response in the supervised text, not the user prompt.

sample = trainer.train_dataset[0]

print("Dataset fields:", sample.keys())

input_ids = sample["input_ids"]
labels = sample["labels"]

supervised_tokens = [
    token_id
    for token_id, label in zip(input_ids, labels)
    if label != -100
]

print(
    "Supervised text:",
    trainer.processing_class.decode(supervised_tokens)
)