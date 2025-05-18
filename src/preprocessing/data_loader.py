import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_dataloaders(config):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["pretrained_model"])

    # Load local text data
    train_texts = load_text_file(config["data"]["train_path"])
    val_texts = load_text_file(config["data"]["val_path"])

    # Load external datasets (e.g., Hugging Face datasets)
    external_train = load_dataset("ag_news", split="train[:10%]")
    external_val = load_dataset("ag_news", split="test[:10%]")

    # Combine local and external datasets
    train_texts.extend([item["text"] for item in external_train])
    val_texts.extend([item["text"] for item in external_val])

    if not train_texts:
        raise ValueError(f"Training data is empty. Check the file at {config['data']['train_path']}")
    if not val_texts:
        raise ValueError(f"Validation data is empty. Check the file at {config['data']['val_path']}")

    logging.info(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples.")

    train_dataset = TextDataset(train_texts, tokenizer, config["training"]["max_seq_length"])
    val_dataset = TextDataset(val_texts, tokenizer, config["training"]["max_seq_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader
