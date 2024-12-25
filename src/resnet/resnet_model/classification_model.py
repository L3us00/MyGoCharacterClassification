import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from ..constants import DEVICE, TRAIN_CONFIG


class ClassificationModel:
    def __init__(self, classification_class: int, pretrained=True):
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, classification_class)
        self.model.to(DEVICE)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, lr: float, epochs: int, save_path: str):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            train_loss = 0
            train_acc = 0
            for batch, (X, y) in enumerate(train_loader):
                X, y = X.to(DEVICE), torch.tensor([int(label) for label in y], dtype=torch.long, device=DEVICE)

                optimizer.zero_grad()
                y_pred = self.model(X)
                loss = loss_fn(y_pred, y)
                acc = (y_pred.argmax(dim=1) == y).float().mean()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc.item()

                if batch % 20 == 0:
                    print(f"Batch {batch}: Loss={loss.item():.4f}, Accuracy={acc.item():.4f}")
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            self.evaluate(val_loader)
        torch.save(self.model.state_dict(), save_path)

    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), torch.tensor([int(label) for label in y], dtype=torch.long, device=DEVICE)
                y_pred = self.model(X)
                loss = nn.CrossEntropyLoss()(y_pred, y)
                acc = (y_pred.argmax(dim=1) == y).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    def load(self, model_path: str):
        # Load the model weights from a saved file
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(DEVICE)

    def predict(self, img_tensor: torch.Tensor):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)

        return output
