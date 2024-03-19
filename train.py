import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from models import ViTB16,ViTS16,FocalNet

train_file_paths, train_labels = load_paths_and_labels("data/split/senpai/train.txt")
test_file_paths, test_labels = load_paths_and_labels("data/split/senpai/test.txt")

train_transform = transforms.Compose([
    GBMTransform(),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 重复通道
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 仿射变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # 随机擦除
    transforms.Lambda(lambda x: x if torch.rand(1) > 0.5 else -x + 1)  # 50%概率随机反转
])

test_transform = transforms.Compose([
    GBMTransform(),  
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

train_dataset = GBMDataset(train_file_paths, train_labels, transform=train_transform)
test_dataset = GBMDataset(test_file_paths, test_labels, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = get_device()

#use different model here
#model = ViTS16().to(device)
model = FocalNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.05)
scheduler = create_scheduler(optimizer)




#6 radom seed

session_folder = create_training_session_folder()

import time

#100epoch
#attention map

#dino on github



num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    start_time = time.time()
    
    save_log(f"Epoch {epoch+1}/{num_epochs} started...", session_folder)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 获取模型的输出
        outputs = model(images)
        #change here if model change
        #logits = outputs.logits
        logits = outputs
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            batches_left = len(train_loader) - batch_idx - 1
            time_per_batch = elapsed_time / (batch_idx + 1)
            remaining_time = batches_left * time_per_batch

            save_log(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}, '
                  f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s', session_folder)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    total_time = time.time() - start_time

    scheduler.step()  # 在每个epoch结束时更新学习率

    current_lr = optimizer.param_groups[0]["lr"]
    
    save_log(f"Epoch {epoch+1} Finished, Loss: {epoch_loss}, Accuracy: {epoch_acc}%, "
                f"Learning Rate: {current_lr}, Total Time: {total_time:.2f}s\n", session_folder)

    save_log(f'Epoch {epoch+1} Finished, Loss: {epoch_loss}, Accuracy: {epoch_acc}%, '
          f'Total Time: {total_time:.2f}s\n', session_folder)



    

    save_model(model, epoch, session_folder)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    save_log(f"Validation Accuracy: {100 * correct / total}%", session_folder)

save_log("Training Complete", session_folder)

