import torch
import torch.nn as nn
import torchvision
from constants import CLASS_NAMES
from evaluation import do_inference


import sys


def predict(image_path):
    model = torchvision.models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    save_path = "saved_models/densenet121/best_model.pth"
    weights = torch.load(f=save_path)
    model.load_state_dict(weights)

    x = do_inference(model, image_path=image_path)
    _, y_pred = torch.max(x, 1)
    pos = y_pred.cpu().numpy()
    print(f"Prediction is: {CLASS_NAMES[pos]}")


if __name__ == "__main__":
    image_path = sys.argv[1]
    print(image_path)
    # image_path = "/Users/samet/Documents/data/TB_Chest_Radiography_Database/Normal/Normal-1821.png"
    predict(image_path)
