import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy(y_true, y_pred):
    """
    Computes the accuracy of the predictions.
    Args:
        y_true (torch.Tensor or np.array): Ground truth labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
    
    Returns:
        float: Accuracy score.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average='binary'):
    """
    Computes the precision of the predictions.
    Args:
        y_true (torch.Tensor or np.array): Ground truth labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str): The method to calculate precision for multi-class classification. ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        float: Precision score.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return precision_score(y_true, y_pred, average=average)


def recall(y_true, y_pred, average='binary'):
    """
    Computes the recall of the predictions.
    Args:
        y_true (torch.Tensor or np.array): Ground truth labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str): The method to calculate recall for multi-class classification. ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        float: Recall score.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return recall_score(y_true, y_pred, average=average)


def f1(y_true, y_pred, average='binary'):
    """
    Computes the F1 score of the predictions.
    Args:
        y_true (torch.Tensor or np.array): Ground truth labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
        average (str): The method to calculate F1 for multi-class classification. ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        float: F1 score.
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average=average)


def cross_entropy_loss(y_true, y_pred):
    """
    Computes the cross-entropy loss between the true labels and predictions.
    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted logits.
    
    Returns:
        torch.Tensor: The cross-entropy loss.
    """
    return F.cross_entropy(y_pred, y_true)


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Computes the binary cross-entropy loss between the true labels and predictions.
    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred (torch.Tensor): Predicted probabilities.
    
    Returns:
        torch.Tensor: The binary cross-entropy loss.
    """
    return F.binary_cross_entropy(y_pred, y_true)


def mean_squared_error(y_true, y_pred):
    """
    Computes the mean squared error between the true labels and predictions.
    Args:
        y_true (torch.Tensor or np.array): Ground truth labels.
        y_pred (torch.Tensor or np.array): Predicted labels.
    
    Returns:
        float: Mean squared error score.
    """
    return torch.mean((y_true - y_pred) ** 2).item()
