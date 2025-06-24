from fastapi import FastAPI, Request
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

base_model = "mistralai/Mistral-7B-Instruct-v0.1"
lora_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return {"response": response}
