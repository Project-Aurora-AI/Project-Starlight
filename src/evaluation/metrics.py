import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy(y_true, y_pred):
    """
    Computes accuracy.
    Args:
        y_true (torch.Tensor or np.array): True labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, average='binary'):
    """
    Computes precision score.
    Args:
        y_true (torch.Tensor or np.array): True labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str, optional): The type of averaging ('binary', 'micro', 'macro', 'weighted'). Default is 'binary'.
    Returns:
        float: Precision score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return precision_score(y_true, y_pred, average=average)

def recall(y_true, y_pred, average='binary'):
    """
    Computes recall score.
    Args:
        y_true (torch.Tensor or np.array): True labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str, optional): The type of averaging ('binary', 'micro', 'macro', 'weighted'). Default is 'binary'.
    Returns:
        float: Recall score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return recall_score(y_true, y_pred, average=average)

def f1(y_true, y_pred, average='binary'):
    """
    Computes F1 score.
    Args:
        y_true (torch.Tensor or np.array): True labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str, optional): The type of averaging ('binary', 'micro', 'macro', 'weighted'). Default is 'binary'.
    Returns:
        float: F1 score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return f1_score(y_true, y_pred, average=average)

def binary_cross_entropy(y_true, y_pred):
    """
    Computes binary cross entropy loss.
    Args:
        y_true (torch.Tensor): True labels (0 or 1).
        y_pred (torch.Tensor): Predicted probabilities (output of sigmoid).
    Returns:
        float: Binary cross entropy loss.
    """
    return F.binary_cross_entropy(y_pred, y_true)

def mean_squared_error(y_true, y_pred):
    """
    Computes mean squared error.
    Args:
        y_true (torch.Tensor or np.array): True labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
    Returns:
        float: Mean squared error.
    """
    if isinstance(y_true, torch.Tensor):
        return F.mse_loss(y_pred, y_true)
    else:
        return ((y_true - y_pred) ** 2).mean()

# Example usage:
if __name__ == "__main__":
    # Example: y_true and y_pred are torch tensors or numpy arrays.
    y_true = torch.tensor([1, 0, 1, 1, 0])
    y_pred = torch.tensor([0.8, 0.1, 0.9, 0.7, 0.2])

    # Assuming we want to use the binary classification version of the metrics
    print("Accuracy:", accuracy(y_true, y_pred.round()))
    print("Precision:", precision(y_true, y_pred.round()))
    print("Recall:", recall(y_true, y_pred.round()))
    print("F1 Score:", f1(y_true, y_pred.round()))
    print("Binary Cross-Entropy:", binary_cross_entropy(y_true.float(), y_pred))
