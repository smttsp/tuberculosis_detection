import os
from datetime import datetime
from enum import Enum

import torch


ROOT_DIR = "/users/samet/Documents/data/TB_Chest_Radiography_Database"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ("Normal", "Tuberculosis")


def get_runtime_str():
    """Getting datetime as a string"""

    runtime_str = (
        datetime.now()
        .isoformat()
        .replace(":", "")
        .replace("-", "")
        .replace("T", "-")
        .split(".")[0]
    )
    return runtime_str


RUNTIME_STR = get_runtime_str()


class DatasetType(str, Enum):
    train = "train"
    val = "val"
    test = "test"
