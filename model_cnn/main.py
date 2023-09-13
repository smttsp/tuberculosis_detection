import argparse
from datetime import datetime

import hydra
import torch
from constants import ROOT_DIR, DatasetType
from dataset import get_loaders_main
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


@hydra.main(
    version_base=None, config_path="./configs", config_name="densenet121"
)
def main(config):
    print(f"TRAINING START TIME: {datetime.now().isoformat()}")

    user_args = config.training
    model, criterion, optimizer, scheduler = densenet121_model(user_args)

    train_loader, test_loader, val_loader = get_loaders_main(
        root_dir=ROOT_DIR,
        split_ratio=user_args.split_ratio,
        batch_size=user_args.batch_size,
    )
    best_model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config=config,
    )
    print(f"TRAINING END TIME: {datetime.now().isoformat()}")
    return best_model


if __name__ == "__main__":
    densenet_121 = main()
    print("finito")
