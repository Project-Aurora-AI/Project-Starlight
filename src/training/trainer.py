import torch
import torch.nn as nn
import torch.optim as optim
from src.training.checkpointing import save_checkpoint
from src.training.metrics import compute_metrics

def train(model, train_loader, val_loader, config, device):
    epochs = config["training"]["epochs"]
    lr = float(config["training"]["learning_rate"])  # Ensure learning rate is a float
    save_path = config["training"].get("checkpoint_path", "models/base_model/last.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            inputs = inputs.float()  # Ensure inputs are of type Float
            inputs = inputs.long()  # Ensure inputs are of type Long for embedding layer
            targets = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")

        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, save_path)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            all_preds.append(outputs)
            all_targets.append(targets)

    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics
