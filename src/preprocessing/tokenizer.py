import os
import json
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, GPT2Tokenizer
from typing import List, Tuple, Dict, Any

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_or_create_tokenizer(vocab_path: str, tokenizer_name: str = "gpt2", vocab_size: int = 50257) -> PreTrainedTokenizerFast:
    """
    Loads an existing tokenizer or creates one from a vocabulary file if it doesn't exist.
    Args:
        vocab_path (str): Path to store or load the tokenizer.
        tokenizer_name (str): Pretrained tokenizer model (e.g., "gpt2").
        vocab_size (int): Desired vocabulary size for the tokenizer.
    
    Returns:
        tokenizer (PreTrainedTokenizerFast): A tokenizer object ready for use.
    """
    if os.path.exists(vocab_path):
        print(f"Loading tokenizer from {vocab_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(vocab_path)
    else:
        print(f"Creating new tokenizer based on {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(vocab_path)

    # Ensure the tokenizer has the correct size
    tokenizer.model_max_length = vocab_size
    return tokenizer

def tokenize_text(text: str) -> List[int]:
    """
    Tokenize a given text using the GPT2 tokenizer.
    Args:
    - text (str): The input text to be tokenized.
    
    Returns:
    - List[int]: Token IDs corresponding to the input text.
    """
    return tokenizer.encode(text, add_special_tokens=True)

def tokenize_data(tokenizer: PreTrainedTokenizerFast, text_data: List[str], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes input text data.
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        text_data (list of str): List of text samples to be tokenized.
        max_length (int): Maximum length of tokens after padding/truncation.
    
    Returns:
        input_ids (list of list of int): Tokenized input ids.
        attention_masks (list of list of int): Attention masks for padding.
    """
    encoding = tokenizer(
        text_data,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask

def save_tokenizer(tokenizer: PreTrainedTokenizerFast, tokenizer_path: str) -> None:
    """
    Saves the tokenizer to a specific directory.
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer object.
        tokenizer_path (str): The path where the tokenizer will be saved.
    """
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """
    Loads a tokenizer from a saved directory.
    Args:
        tokenizer_path (str): Path where the tokenizer is saved.
    
    Returns:
        tokenizer (PreTrainedTokenizerFast): The loaded tokenizer.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded from {tokenizer_path}")
    return tokenizer

def process_and_save_data(tokenizer: PreTrainedTokenizerFast, data_path: str, save_path: str, max_length: int = 512) -> None:
    """
    Tokenizes raw data and saves the processed tokenized data as JSON.
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
        data_path (str): Path to raw data (e.g., text files or JSON).
        save_path (str): Path where the processed data will be saved.
        max_length (int): Maximum token length for each sequence.
    """
    try:
        with open(data_path, 'r') as file:
            raw_data = json.load(file)  # Assuming the data is in JSON format
    except Exception as e:
        print(f"Error loading raw data from {data_path}: {e}")
        return

    # Assuming each item has a 'text' field
    text_data = [item['text'] for item in raw_data if 'text' in item]
    if not text_data:
        print("No 'text' field found in the data.")
        return

    # Tokenize the data
    input_ids, attention_mask = tokenize_data(tokenizer, text_data, max_length)

    # Save the tokenized data as JSON
    processed_data = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist()
    }

    try:
        with open(save_path, 'w') as save_file:
            json.dump(processed_data, save_file)
        print(f"Processed data saved to {save_path}")
    except Exception as e:
        print(f"Error saving processed data to {save_path}: {e}")
