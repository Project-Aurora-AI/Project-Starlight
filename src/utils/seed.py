import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set the seed for reproducibility. This function ensures that random processes in 
    various libraries (PyTorch, NumPy, Python's random) are initialized with the same seed.
    
    Args:
        seed (int): The seed value to set.
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (helps in reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility.")

