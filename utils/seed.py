import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Pin all RNG sources for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops — slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
