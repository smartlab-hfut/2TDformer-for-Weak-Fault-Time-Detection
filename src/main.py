import torch
import torch.nn as nn
import torch.optim as optim

from train1 import (
    prepare_dataset,
    train_one_epoch,
    evaluate,
)
from model import Transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # ======================
    # 路径与超参数
    # ======================
    data_path = "../data/"
    batch_size = 16
    lr = 0.001
    num_epochs = 50
    dimension = 6
    num_heads = 6
    d_hid = 400
    d_inner = 400
    n_layers = 1
    dropout = 0
    mul_dim = 3
    step_size = 50

    # ======================
    # 加载数据
    # ======================
    train_loader, test_loader = prepare_dataset(
        data_path, batch_size=batch_size
    )

    # ======================
    # 创建模型
    # ======================
    model = Transformer(
        dimension=dimension,
        num_heads=num_heads,
        d_hid=d_hid,
        d_inner=d_inner,
        n_layers=n_layers,
        dropout=dropout,
        mul_dim=mul_dim,
        step_size=step_size,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ======================
    # 训练
    # ======================
    print("===== Start Training =====")

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        evaluate(model, test_loader, criterion, device, epoch)

    print("===== Finished =====")



