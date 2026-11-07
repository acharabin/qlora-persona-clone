import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import config_from_cli

config = config_from_cli()
inference = config["inference"]
BASE_MODEL = config["model"]["id"]
ADAPTER_PATH = inference["adapter_path"]
device = inference["device"]
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
precision = "cuda" if device.startswith("cuda") else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=getattr(torch, inference[f"{precision}_dtype"]),
    device_map=inference[f"{precision}_device_map"] if device in ("cuda", "cpu") else device,
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

inputs = tokenizer.apply_chat_template(
    inference["messages"],
    **inference["chat_template"],
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        **inference["generation"],
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=inference["skip_special_tokens"],
)

print(response)