import os

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from .constants import CLASS_NAMES, MODEL_PATH
from .dataset import get_test_transform


def do_inference(model, image_path):
    transform = get_test_transform()
    image = Image.open(image_path).convert("RGB")

    input_data = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_data)

    return output


def get_full_path(path):
    if "~" in path:
        home_dir = os.path.expanduser("~")
        path = path.replace("~", home_dir)

    return path


def get_model_prediction(image_path, model_path=MODEL_PATH):
    model = torchvision.models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    path = get_full_path(model_path)

    weights = torch.load(f=path)
    model.load_state_dict(weights)

    x = do_inference(model, image_path=image_path)
    _, y_pred = torch.max(x, 1)
    pos = y_pred.cpu().numpy()[0]
    print(pos)
    print(f"Prediction is: {CLASS_NAMES[pos]}")
    return CLASS_NAMES[pos]
