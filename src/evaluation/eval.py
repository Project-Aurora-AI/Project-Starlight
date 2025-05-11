import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.model.model import MyModel  # Replace with your actual model class
from data.dataset import MyDataset  # Replace with your actual dataset class
from src.evaluation.metrics import accuracy, precision, recall, f1

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return {
        "accuracy": accuracy(all_labels, all_preds),
        "precision": precision(all_labels, all_preds, average='macro'),
        "recall": recall(all_labels, all_preds, average='macro'),
        "f1": f1(all_labels, all_preds, average='macro'),
    }

def load_model(checkpoint_path, device):
    model = MyModel()  # Replace with your model’s init
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model

def run_evaluation(data_path, checkpoint_path, batch_size=32, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset(data_path, split='val')  # or 'test'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = load_model(checkpoint_path, device)

    results = evaluate(model, dataloader, device)

    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    run_evaluation(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
