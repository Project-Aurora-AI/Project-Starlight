from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Init FastAPI app
app = FastAPI(title="Jade AI Inference API", version="1.0.0")

# Load model & tokenizer
MODEL_NAME = "your_model_path_or_repo"  # Replace with your actual local dir or HF repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Request model
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    chittersync_user_id: str = None  # Optional ChitterSync ID for context

# Response model
class GenerateResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    # Tokenize input
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cut off input from output
    response_text = generated[len(req.prompt):].strip()

    # Optionally log or use `chittersync_user_id` for analytics/user tuning later
    return GenerateResponse(response=response_text)
