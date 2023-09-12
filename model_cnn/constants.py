import os

import torch


ROOT_DIR = "/users/samet/Documents/data/TB_Chest_Radiography_Database"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
