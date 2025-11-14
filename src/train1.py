import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

from model import Transformer



def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



def normalize_data(data):
    scaler = MinMaxScaler()
    out = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            out[i, :, j] = scaler.fit_transform(data[i, :, j].reshape(-1, 1)).ravel()

    return out


def standardize_data(data):
    scaler = StandardScaler()
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    flat = scaler.fit_transform(flat)
    return flat.reshape(shape)



def load_dataset(path, is_train=True):
    mode = "train" if is_train else "test"

    data = pd.read_csv(f"{path}/{mode}_data2.csv", header=None).values.astype(float)
    label = pd.read_csv(f"{path}/{mode}_label2.csv", header=None).values.astype(int)

    length = len(data) // 400
    data = data.reshape(length, 400, -1)
    label = label.reshape(length, 400)

    return data, label


def prepare_dataset(path, batch_size=16, test_ratio=0.2):
    data, label = load_dataset(path, is_train=True)

    data = normalize_data(data)
    data = standardize_data(data)

    train_x, test_x, train_y, test_y = train_test_split(
        data, label, test_size=test_ratio, random_state=42
    )

    train_loader = DataLoader(MyDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    start = time.time()

    for x, y in loader:
        x, y = x.to(device), y.float().to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = (torch.sigmoid(out) > 0.5).float()
        total_correct += (pred == y).float().sum().item()
        total_samples += y.numel()

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples
    print(f"Epoch {epoch+1} Train | Loss={avg_loss:.4f}, Acc={acc:.4f}, Time={time.time()-start:.2f}s")


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).float()
            total_correct += (pred == y).float().sum().item()
            total_samples += y.numel()

    avg_loss = total_loss / len(loader)
    acc = total_correct / total_samples
    print(f"Epoch {epoch+1} Test  | Loss={avg_loss:.4f}, Acc={acc:.4f}")

    return avg_loss, acc




