import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model, optimizer state_dict, epoch, and loss to a checkpoint file.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        loss (float): The loss at the current epoch.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): The name of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location='cpu'):
    """
    Loads the model, optimizer state_dict, epoch, and loss from a checkpoint file.
    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state_dict into.
        checkpoint_path (str): The path to the checkpoint file.
        map_location (str): Location to load the checkpoint (default is 'cpu').
    
    Returns:
        epoch (int): The epoch from which training can resume.
        loss (float): The loss from the checkpoint.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch} with loss {loss}.")
        
        return epoch, loss
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return None, None


def resume_training(model, optimizer, checkpoint_path, map_location='cpu'):
    """
    Resumes training from a checkpoint if available.
    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state_dict into.
        checkpoint_path (str): The path to the checkpoint file.
        map_location (str): Location to load the checkpoint (default is 'cpu').
    
    Returns:
        epoch (int): The epoch to resume training from.
        loss (float): The loss from the checkpoint.
    """
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path, map_location)
    if epoch is None:
        print("No checkpoint found, starting from scratch.")
        epoch = 0
        loss = None

    return epoch, loss
