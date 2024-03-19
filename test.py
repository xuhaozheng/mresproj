import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utils import GBMDataset, GBMTransform, load_paths_and_labels, get_device
from models import ViTS16


model_weights_path = "/home/neuralmaster/Desktop/MRes/YKLi/mresproj/checkpoints/ViT/model_epoch_82_best.pth"

test_file_paths, test_labels = load_paths_and_labels("data/split/senpai/test.txt")

test_transform = transforms.Compose([
    GBMTransform(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

test_dataset = GBMDataset(test_file_paths, test_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = get_device()

model = ViTS16().to(device)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
specificity = tn / (tn+fp)

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")
