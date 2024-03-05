import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class GBMDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 加载图像并转换为灰度
        image = Image.open(self.file_paths[idx]).convert('L')
        # 应用转换（如果有的话）
        if self.transform:
            image = self.transform(image)
        # 获取对应的标签
        label = self.labels[idx]
        return image, label

class GBMTransform(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, img):
        # 将PIL图像转换为Tensor
        img = transforms.functional.to_tensor(img)

        _, H, W = img.shape
        # 裁剪成最大正方形
        if H != W:
            # 计算裁剪的起始点和结束点，以确保裁剪区域保留最中心的部分
            start = (H - W) // 2
            # 对于高度大于宽度的图像，裁剪出中心的正方形
            if start >= 0:
                img = img[:, start:start+W, :]
            else:
                # 对于宽度大于高度的图像，裁剪出中心的正方形
                start = (W - H) // 2
                img = img[:, :, start:start+H]
        
        # 先调整图像大小
        img = transforms.functional.resize(img, [self.output_size, self.output_size])

        # 之后创建与图像大小一致的掩码
        _, height, width = img.size()
        center = torch.tensor([width / 2, height / 2])
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= (min(width, height) / 2)
        
        # 应用掩码
        img = img * mask.float().unsqueeze(0)  # 为掩码增加一个批处理维度以匹配img的维度

        return img






def load_paths_and_labels(txt_file):

    image_paths = []
    labels = []
    
    with open(txt_file, 'r') as file:
        for line in file:
            path, label = line.strip().split(' ')
            path = "data/image/" + path
            image_paths.append(path)
            labels.append(int(label))
    
    return image_paths, labels


# 定义最终的transform
transform = transforms.Compose([
    GBMTransform(),  # 圆形mask预处理并调整大小
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# 假设file_paths是包含图像文件路径的列表，labels是对应的标签列表
file_paths, labels = load_paths_and_labels("split/senpai/1.txt")

dataset = GBMDataset(file_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import matplotlib.pyplot as plt
import torch

# 使用dataloader在训练循环中
for images, labels in dataloader:
    # 假设images是一个批次的图像张量，大小为 [batch_size, channels, height, width]
    
    # 将Tensor转换为NumPy数组，选择要显示的图像索引（这里我们选择批次中的第一个图像）
    image_to_show = images[0].numpy()
    
    # 对于单通道图像（如黑白图），需要调整形状为 [height, width]
    if image_to_show.shape[0] == 1:  # 单通道图像
        image_to_show = image_to_show.squeeze(0)  # 从 [1, height, width] 转换为 [height, width]
    
    # 使用matplotlib显示图像
    plt.imshow(image_to_show, cmap='gray')  # 对于黑白图像使用灰度色图
    plt.title(f"Label: {labels[0]}")  # 显示图像对应的标签
    plt.show()
    
    # 这里进行模型的训练，images和labels已经准备好
    # 为了演示，我们只显示第一个批次的第一张图像，然后跳出循环
    break
