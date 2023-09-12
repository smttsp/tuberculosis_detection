import argparse
import time

import cv2
import matplotlib.pyplot as plt
import torch


torch.manual_seed(1)

ROOT_DIR = "/users/samet/Documents/data/TB_Chest_Radiography_Database"


def parse_args(root_dir=ROOT_DIR):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
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
        "--data_path", type=str, default=root_dir, help="Path to data."
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


def main():
    print(f"TRAINING START TIME: {time.time()}")
    args = parse_args()


if __name__ == "__main__":
    im = cv2.imread(f"{ROOT_DIR}/Normal/Normal-100.png")
    plt.imshow(im)
    main()
