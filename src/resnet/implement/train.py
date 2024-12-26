import torch
from src.resnet.resnet_model.classification_model import ClassificationModel
from ..utils.get_dataloader import get_dataloader
from ..constants import TRAIN_CONFIG


def train():
    torch.manual_seed(TRAIN_CONFIG.get("random_seed",42))
    train_loader, val_loader = get_dataloader()

    model_resnet = ClassificationModel(TRAIN_CONFIG["classification_class"])
    model_resnet.train(train_loader=train_loader,
                       val_loader=val_loader,
                       lr=TRAIN_CONFIG["lr"],
                       epochs=TRAIN_CONFIG["epochs"],
                       save_path=TRAIN_CONFIG["model_path"])
