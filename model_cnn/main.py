import argparse
from datetime import datetime
import torch
from constants import ROOT_DIR
from dataset import ChestXRayDataset, get_loaders, get_transformation
from model import densenet121_model
from training import train_model


torch.manual_seed(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularisation")
    parser.add_argument(
        "--aug",
        action="store_true",
        default=False,
        help="Use data augmentation",
    )
    parser.add_argument(
        "--data_path", type=str, default=ROOT_DIR, help="Path to data."
    )
    parser.add_argument(
        "--bond_dim", type=int, default=5, help="MPS Bond dimension"
    )
    parser.add_argument(
        "--nChannel", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--dense_net",
        action="store_true",
        default=False,
        help="Using Dense Net model",
    )
    user_args = parser.parse_args([])
    return user_args


def main():
    print(f"TRAINING START TIME: {datetime.now().isoformat()}")
    user_args = parse_args()

    model, criterion, optimizer, scheduler = densenet121_model(user_args)

    train_transform, valid_transform = get_transformation()
    dataset = ChestXRayDataset(root_dir=ROOT_DIR, transform=train_transform)

    train_loader, test_loader, val_loader = get_loaders(
        dataset, split_ratios=[0.7, 0.15, 0.15], batch_size=user_args.batch_size
    )
    densenet_121 = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        user_args=user_args,
    )
    print(f"TRAINING END TIME: {datetime.now().isoformat()}")

    return densenet_121


if __name__ == "__main__":
    densenet_121 = main()
    print("finito")
