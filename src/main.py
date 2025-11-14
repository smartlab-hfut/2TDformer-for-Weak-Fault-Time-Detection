import argparse
from train import prepare_dataset, train_one_epoch, evaluate
from model import Transformer
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2TDformer Model")

    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset folder")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_hid", type=int, default=400)
    parser.add_argument("--d_inner", type=int, default=400)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mul_dim", type=int, default=3)
    parser.add_argument("--step_size", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load data
    train_loader, test_loader = prepare_dataset(
        args.data_path,
        batch_size=args.batch_size
    )

    # Build model
    model = Transformer(
        dimension=args.dimension,
        num_heads=args.num_heads,
        d_hid=args.d_hid,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        dropout=args.dropout,
        mul_dim=args.mul_dim,
        step_size=args.step_size
    ).to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("===== Start Training =====")

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        evaluate(model, test_loader, criterion, device, epoch)

    print("===== Finished =====")
