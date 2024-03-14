import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_device():
    if torch.backends.mps.is_available():  # 检查是否支持Apple Metal
        return torch.device("mps")
    elif torch.cuda.is_available():  # 检查CUDA是否可用
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    


def create_scheduler(optimizer, warmup_epochs=10, num_epochs=100, base_lr=1e-3, min_lr=1e-5, init_lr=1e-6):
    def warmup_cosine_annealing_scheduler(epoch):
        if epoch < warmup_epochs:
            return init_lr + (base_lr - init_lr) * (epoch / warmup_epochs)
        else:
            cos = np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + cos)

    return LambdaLR(optimizer, lr_lambda=warmup_cosine_annealing_scheduler)