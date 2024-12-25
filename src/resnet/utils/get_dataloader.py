from src.resnet.dataloader.mygo_dataloader import dataset_transform, data_loader
from ..utils.import_csv import import_csv,split_data
from typing import Tuple
from torch.utils.data import DataLoader
from ..constants import TRAIN_CONFIG

def get_dataloader() -> Tuple[DataLoader, DataLoader]:
    data = import_csv()
    train_data, val_data = split_data(data=data)
    train_data_transformed = dataset_transform(data=train_data, mode="train")
    val_data_transformed = dataset_transform(data=val_data, mode="val")
    train_loader = data_loader(tranformed_dataset=train_data_transformed, batch_size=TRAIN_CONFIG.get("batch",16), num_workers=1, shuffle=True)
    val_loader = data_loader(tranformed_dataset=val_data_transformed, batch_size=TRAIN_CONFIG.get("batch",16), num_workers=1, shuffle=True)
    return train_loader,val_loader