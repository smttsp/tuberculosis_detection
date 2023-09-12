import pandas
import torch
from sklearn.metrics import confusion_matrix


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

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    f1score = (2 * TP) / (2 * TP + FN + FP)
    evaluate_list = [[accuracy, precision, sensitivity, f1score, specificity]]
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
