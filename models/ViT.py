from transformers import ViTForImageClassification, ViTConfig
import torch

class ViTB16(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(ViTB16, self).__init__()
        configuration = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels)
        self.vit = ViTForImageClassification(configuration)

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values)
    
class ViTS16(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(ViTS16, self).__init__()
        configuration = ViTConfig.from_pretrained("WinKawaks/vit-small-patch16-224", num_labels=num_labels) 
        self.vit = ViTForImageClassification(configuration)

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values)


