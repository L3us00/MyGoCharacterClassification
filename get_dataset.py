from src.resnet.utils.modify_dataset import get_dataset
from src.resnet.utils.label_to_csv import to_csv


data = get_dataset(100)
to_csv(data)
