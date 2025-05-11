import os
import torch
import random
import argparse
import yaml
from src.model.model import build_model
from src.training.trainer import train
from src.utils.logger import setup_logging
from src.preprocessing.data_loader import get_dataloaders
from src.utils.seed import set_seed

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)

    # Setup
    set_seed(config["training"]["seed"])
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_dataloaders(config)

    # Build model
    model = build_model(config).to(device)

    # Start training
    train(model, train_loader, val_loader, config, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Project Starlight")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)
