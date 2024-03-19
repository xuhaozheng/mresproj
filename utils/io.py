import torch
import os
from datetime import datetime

    

def create_training_session_folder(root_dir="checkpoints"):
    os.makedirs(root_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_folder = os.path.join(root_dir, f"session_{timestamp}")
    os.makedirs(session_folder)
    return session_folder

def save_model(model, epoch, save_dir):
    filename = f"model_epoch_{epoch+1}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)

def save_log(msg, session_folder):
    log_file = os.path.join(session_folder, 'log.txt')
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


