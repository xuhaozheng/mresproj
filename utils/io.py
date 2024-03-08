import torch
import os
from datetime import datetime

def get_device():
    if torch.backends.mps.is_available():  # 检查是否支持Apple Metal
        return torch.device("mps")
    elif torch.cuda.is_available():  # 检查CUDA是否可用
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def create_training_session_folder(root_dir="checkpoints"):
    os.makedirs(root_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_folder = os.path.join(root_dir, f"session_{timestamp}")
    os.makedirs(session_folder)
    return session_folder

def save_model_state(model, epoch, save_dir):
    filename = f"model_epoch_{epoch+1}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)


