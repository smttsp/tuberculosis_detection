import argparse
from datetime import datetime

import hydra
import torch
from constants import ROOT_DIR
from dataset import get_loaders_main
from model import densenet121_model
from training import train_model


torch.manual_seed(0)


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
