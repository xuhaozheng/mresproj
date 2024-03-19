import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

from utils import *
from models import ViTB16, ViTS16, FocalNet





def main(rank, world_size):
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=128, sampler=test_sampler)

    model = FocalNet().to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05)
    scheduler = create_scheduler(optimizer)

    if rank == 0:
        session_folder = create_training_session_folder()
    
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        start_time = time.time()  # 记录epoch开始时间
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:  # 每处理10个batch计算一次时间
                elapsed_time = time.time() - start_time  # 已过时间
                total_batches = len(train_loader)
                batches_left = total_batches - batch_idx - 1
                avg_time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_time = avg_time_per_batch * batches_left  # 剩余时间
                
                if rank == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, '
                          f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s')
                    save_log(f'Epoch: {epoch+1}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, '
                             f'Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s', session_folder)
        
# 每个epoch结束后进行测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 在所有进程上累计测试损失和正确的数量
        test_loss = torch.tensor(test_loss).to(rank)
        correct = torch.tensor(correct).to(rank)
        total = torch.tensor(total).to(rank)
        dist.reduce(test_loss, dst=0)
        dist.reduce(correct, dst=0)
        dist.reduce(total, dst=0)

        # 只在rank 0进程上打印测试结果
        if rank == 0:
            avg_loss = test_loss / len(test_loader)
            accuracy = 100. * correct / total
            save_log(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)', session_folder)


        if rank == 0:
            total_epoch_time = time.time() - start_time
            save_log(f"Epoch {epoch+1} Finished, Total Time: {total_epoch_time:.2f}s", session_folder)
            save_model(model, epoch, session_folder)

    if rank == 0:
        save_log("Training Complete", session_folder)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # 假设您有2个GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
