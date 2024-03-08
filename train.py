import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils import GBMDataset,GBMTransform,load_paths_and_labels,get_device
from utils import *
from models import ViTB16,ViTS16

train_file_paths, train_labels = load_paths_and_labels("data/split/senpai/train.txt")
test_file_paths, test_labels = load_paths_and_labels("data/split/senpai/test.txt")

train_transform = transforms.Compose([
    GBMTransform(),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
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

model = ViTB16().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#AdamW 0.05 learning rate
#cos learing scal warmup epoch:10 initial learning rate 10-6 base learning rate 10-3 min learning rate 10-5
#6 radom seed

session_folder = create_training_session_folder()

import time

#100epoch
#attention map

#dino on github



num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{num_epochs} started...")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 获取模型的输出
        outputs = model(images)
        logits = outputs.logits
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

            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}, '
                  f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    total_time = time.time() - start_time

    print(f'Epoch {epoch+1} Finished, Loss: {epoch_loss}, Accuracy: {epoch_acc}%, '
          f'Total Time: {total_time:.2f}s\n')



    
    # 保存模型状态
    save_model_state(model, epoch, session_folder)

    # 验证阶段
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
    print(f"Validation Accuracy: {100 * correct / total}%")

print("Training Complete")

