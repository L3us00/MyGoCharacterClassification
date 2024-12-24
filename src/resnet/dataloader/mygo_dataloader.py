import os
import cv2
import torch
from PIL import Image
from typing import List, Dict, Tuple
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, data: List[Dict], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image_path = self.data[idx]["file_path"]
        label = self.data[idx]["class"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        print(type(image))
        return image, label


def dataset_transform(data: List[Dict], mode: str) -> CustomImageDataset:
    transform = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomApply([
            transforms.RandomRotation(30),
        ], p=0.3),

        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ], p=0.3),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3),
        ], p=0.3),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),

    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    }
    
    return CustomImageDataset(data=data, transform=transform[mode])

def data_loader(tranformed_dataset: CustomImageDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset=tranformed_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)