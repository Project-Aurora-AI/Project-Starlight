import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(prompt, model, tokenizer, max_tokens=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt):].strip()

def main():
    parser = argparse.ArgumentParser(description="Jade Inference CLI")
    parser.add_argument("prompt", type=str, help="Prompt to send to the model")
    parser.add_argument("--model", type=str, default="your_model_path_or_repo", help="Path or name of the model")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    response = generate_response(
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"{response}")

if __name__ == "__main__":
    main()
