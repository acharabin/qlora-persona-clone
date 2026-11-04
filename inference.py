import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-8B"
# ADAPTER_PATH = "./models/adapter"
# ADAPTER_PATH = "outputs/qwen3-persona-v1/adapter"
ADAPTER_PATH = "outputs/qwen3-persona-v1/checkpoints/checkpoint-30"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else "cpu",
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

messages = [
    {
        "role": "user",
        # "content": "What advice would you give someone considering a career change?",
        "content": "What's the most important thing to you?",

    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print(response)