import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class Cat_Dog_Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    
    def __getitem__(self, idx):
        data = Image.open(os.path.join(self.path, self.file_list[idx]))
        data = self.transform(data)
        data = np.array(data)
        label = 1 if "cat" in self.file_list[idx] else 0
        return data, label
    
    def __len__(self):
        return len(self.file_list)

def create_dataloader(is_trian=True, batch_size=32):
    if is_trian:
        dataset = Cat_Dog_Dataset('train')
        data_loader = DataLoader(dataset, batch_size, shuffle=True)
    else:
        dataset = Cat_Dog_Dataset('val')
        data_loader = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader

    
    
