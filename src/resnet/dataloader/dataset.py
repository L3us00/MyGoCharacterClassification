import cv2
import torch
from PIL import Image
from typing import List, Dict, Tuple
from torch.utils.data import Dataset


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
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        print(type(image))
        return image, label