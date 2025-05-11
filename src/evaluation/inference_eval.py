import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict, Any

def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  tokenizer: PreTrainedTokenizer,
                  device: torch.device,
                  postprocess_fn: Callable[[Any], str] = None) -> List[Dict[str, str]]:
    """
    Run inference using a trained model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for evaluation data.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding model outputs.
        device (torch.device): Device to run the model on.
        postprocess_fn (Callable, optional): Optional function to postprocess model output.

    Returns:
        List[Dict[str, str]]: List of predictions and corresponding inputs.
    """
    model.eval()
    model.to(device)
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

            decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            if postprocess_fn:
                decoded_outputs = [postprocess_fn(output) for output in decoded_outputs]

            for inp, out in zip(decoded_inputs, decoded_outputs):
                results.append({"input": inp, "output": out})

    return results

def evaluate_predictions(predictions: List[Dict[str, str]], reference_key: str = "reference") -> None:
    """
    Print predictions and optionally compare with references.

    Args:
        predictions (List[Dict[str, str]]): List of prediction dictionaries with keys "input", "output", and optionally "reference".
        reference_key (str): Key to use for reference output if available.

    Returns:
        None
    """
    for i, pred in enumerate(predictions):
        print(f"[{i}] Input: {pred['input']}")
        print(f"    Output: {pred['output']}")
        if reference_key in pred:
            print(f"    Reference: {pred[reference_key]}")
        print("-" * 50)
