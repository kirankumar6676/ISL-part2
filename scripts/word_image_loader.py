import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Training transforms with augmentation
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transform for testing/inference
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class WordImageDataset(Dataset):

    def __init__(self, root, augment=True):
        self.samples = []
        # Only include directories (skip files like dataset_summary.txt)
        self.labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.map = {l: i for i, l in enumerate(self.labels)}
        self.augment = augment

        for label in self.labels:
            folder = os.path.join(root, label)
            for img in os.listdir(folder):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        
        # Use augmented transform for training
        if self.augment:
            img = train_transform(img)
        else:
            img = transform(img)

        return img, torch.tensor(self.map[label])
