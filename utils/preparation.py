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
        image = Image.open(self.file_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class GBMTransform(object):
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, img):
        img = transforms.functional.to_tensor(img)

        _, H, W = img.shape
        if H != W:
            start = (H - W) // 2
            if start >= 0:
                img = img[:, start:start+W, :]
            else:
                start = (W - H) // 2
                img = img[:, :, start:start+H]
        
        img = transforms.functional.resize(img, [self.output_size, self.output_size])

        _, height, width = img.size()
        center = torch.tensor([width / 2, height / 2])
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        dist_from_center = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= (min(width, height) / 2)
        

        img = img * mask.float().unsqueeze(0) 

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



transform = transforms.Compose([
    GBMTransform(),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])


file_paths, labels = load_paths_and_labels("split/senpai/1.txt")

dataset = GBMDataset(file_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import matplotlib.pyplot as plt
import torch


for images, labels in dataloader:

    image_to_show = images[0].numpy()
    

    if image_to_show.shape[0] == 1:  
        image_to_show = image_to_show.squeeze(0) 
    
    plt.imshow(image_to_show, cmap='gray')  
    plt.title(f"Label: {labels[0]}")  
    plt.show()

    break
