import os
from typing import Any
from ..constants import DATASET_DIR,DATACLASS


def get_dataset(image_needs:int) -> Any:
    dataset = []
    for char_idx in DATACLASS:
        dir_path = os.path.join(DATASET_DIR,DATACLASS[char_idx])
        images = [{"file_path": os.path.join(dir_path, file), "class": char_idx} 
              for file in os.listdir(dir_path) if file.lower().endswith('.jpg')]
        stride = int(len(images)/ image_needs)
        if stride == 0:
            dataset += images
            continue
        temp = [images[idx*stride] for idx in range(0,image_needs)]
        dataset += temp
    return dataset




