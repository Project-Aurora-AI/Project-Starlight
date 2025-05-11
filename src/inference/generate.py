import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextGenerator:
    def __init__(self, model_path: str, device: str = None):
        """
        Initializes the text generation pipeline.

        Args:
            model_path (str): Path to the model directory or HuggingFace model repo.
            device (str, optional): Device to run on (e.g., "cuda", "cpu"). If None, auto-select.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generates text from a prompt.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Max number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p (nucleus) sampling.
            do_sample (bool): Whether to use sampling; if False, uses greedy decoding.

        Returns:
            str: Generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text[len(prompt):].strip()
