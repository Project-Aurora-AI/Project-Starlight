import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path):
    # Ensure the path includes a file name
    if os.path.isdir(path):
        path = os.path.join(path, f"checkpoint_epoch_{epoch}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
