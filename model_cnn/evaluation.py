import pandas
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix

from .constants import CLASS_NAMES
from .dataset import get_test_transform


def get_evaluation_metrics(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    precision = tp / (tp + fp)
    f1score = (2 * tp) / (2 * tp + fn + fp)
    evaluate_list = [[accuracy, precision, sensitivity, f1score, specificity]]
    return evaluate_list


def pytorch_predict(model, test_loader, device):
    """
    Make prediction from a pytorch model
    """

    # set model to evaluate model
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)

            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()

    evaluate_list = get_evaluation_metrics(y_pred, y_true)
    df = pandas.DataFrame(
        evaluate_list,
        columns=[
            "Accuracy",
            "Precision",
            "Sensitivity",
            "F1Score",
            "Specificity",
        ],
        dtype=float,
    )
    return df


def do_inference(model, image_path):
    transform = get_test_transform()
    image = Image.open(image_path).convert("RGB")

    input_data = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_data)

    return output


def get_model_prediction(image_path):
    model = torchvision.models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 2), nn.Sigmoid())

    save_path = "model_cnn/saved_models/densenet121/best_model.pth"
    weights = torch.load(f=save_path)
    model.load_state_dict(weights)

    x = do_inference(model, image_path=image_path)
    _, y_pred = torch.max(x, 1)
    pos = y_pred.cpu().numpy()[0]
    print(pos)
    print(f"Prediction is: {CLASS_NAMES[pos]}")
