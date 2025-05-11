import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

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

    # Load text data
    train_texts = load_text_file(config["data"]["train_path"])
    val_texts = load_text_file(config["data"]["val_path"])

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
