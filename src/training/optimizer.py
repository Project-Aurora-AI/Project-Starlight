import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

def get_optimizer(model, config):
    """
    Sets up the optimizer for the model based on the configuration.
    Default: AdamW with weight decay for transformer models.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-8,  # commonly used for transformers
        weight_decay=0.01  # regularization term
    )
    return optimizer

def get_scheduler(optimizer, config):
    """
    Returns a learning rate scheduler for adjusting learning rate during training.
    """
    scheduler = StepLR(
        optimizer,
        step_size=1,  # step every epoch (can adjust)
        gamma=0.95  # decay factor
    )
    return scheduler
