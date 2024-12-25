from typing import List, Dict
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import CustomImageDataset


def dataset_transform(data: List[Dict], mode: str) -> CustomImageDataset:
    transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.RandomRotation(30),
        ], p=0.3),

        # transforms.RandomApply([
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # ], p=0.3),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3),
        ], p=0.3),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),

    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    }
    return CustomImageDataset(data=data, transform=transform[mode])

def data_loader(
        tranformed_dataset: CustomImageDataset, 
        batch_size: int, 
        num_workers: int, 
        shuffle: bool
        ) -> DataLoader:
    return DataLoader(dataset=tranformed_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)