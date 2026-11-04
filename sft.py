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

# Set path

ROOT = Path("/workspace/qlora-persona-clone/qlora-persona-clone")
OUTPUT_DIR = ROOT / "outputs" / "qwen3-persona-v1"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" # Load the dataset

# Load from Hugging Face
dataset = load_dataset("acharabin/AC-persona",data_files="persona_lora_authentic_only_fixed.jsonl",split="train")

# Load from Local
# dataset = load_dataset("json",data_files="./AC-persona/persona_lora_authentic_only_fixed.jsonl",split="train")

# Partition the dataset into train & eval

splits = dataset.train_test_split(
    test_size=0.2,
    seed=42,
)

train_dataset = splits["train"]
eval_dataset = splits["test"]

print(train_dataset[0])

# Model ID

model_id = "Qwen/Qwen3-8B"

# Instantiate Model

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
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2"
    # Reverting to sdpa as flash attention introduces package version dependencies
    attn_implementation="sdpa"
)

# Set Training Config

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),

    # Training
    num_train_epochs=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    max_length=512,

    # Dataset processing
    assistant_only_loss=True,

    # packing=True,
    # packing_strategy="bfd",

    # Reverting for initial trial
    packing=False,
    padding_free=False,

    # Monitoring
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    report_to="tensorboard",
    run_name="qwen3-persona-v1",

    # Evaluation and checkpoints
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,

    # Below not accepted in package version, will retain thinking tokens which can be dropped at inference time
    # chat_template_kwargs={"enable_thinking": False}
)

# print(training_args.per_device_train_batch_size)
# print(training_args.gradient_accumulation_steps)

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

last_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))

print("Resuming from:", last_checkpoint)

trainer.train(resume_from_checkpoint=last_checkpoint)

# Evaluate the best checkpoint
metrics = trainer.evaluate()
print(metrics)

# Save the LoRA adapter
trainer.save_model(
    "/workspace/qlora-persona-clone/outputs/qwen3-persona-v1/adapter"
)