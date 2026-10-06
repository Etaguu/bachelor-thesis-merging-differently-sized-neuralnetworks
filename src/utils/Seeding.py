import random
import numpy as np
import torch

def set_seed(seed):
    """
    Makes all random processes determinstic by making them dependent on a seed
    Ensures reproducibility across all used frameworks and Python

    Args:
        seed: integer seed value

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
