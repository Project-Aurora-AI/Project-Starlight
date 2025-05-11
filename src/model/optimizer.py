import torch
from torch import optim

def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0):
    """
    Creates and returns an optimizer for the model.
    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_name (str): The type of optimizer to use. Options: 'adam', 'sgd', 'adamw'.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 regularization) parameter.
    
    Returns:
        torch.optim.Optimizer: The optimizer for the model.
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported. Choose from 'adam', 'sgd', or 'adamw'.")
    
    print(f"Optimizer {optimizer_name} created with learning rate {lr} and weight decay {weight_decay}")
    return optimizer


def get_scheduler(optimizer, scheduler_name='steplr', step_size=10, gamma=0.1):
    """
    Creates and returns a learning rate scheduler.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for.
        scheduler_name (str): The type of scheduler. Options: 'steplr', 'reduceonplateau'.
        step_size (int): The step size for learning rate decay (for StepLR).
        gamma (float): The decay factor (for StepLR).
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
    """
    if scheduler_name.lower() == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'reduceonplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=5, verbose=True)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not supported. Choose from 'steplr' or 'reduceonplateau'.")
    
    print(f"Learning rate scheduler {scheduler_name} created with step_size {step_size} and gamma {gamma}")
    return scheduler
