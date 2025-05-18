import torch
import torch.nn.functional as F

def compute_metrics(predictions, targets):
    # Initialize accumulators for metrics
    total_correct = 0
    total_count = 0
    total_nll = 0

    for preds, trues in zip(predictions, targets):
        # Get top predictions
        pred_ids = preds.argmax(dim=-1)

        # Accuracy
        correct = (pred_ids == trues).float().sum().item()
        total_correct += correct
        total_count += trues.numel()

        # Perplexity
        with torch.no_grad():
            log_probs = F.log_softmax(preds, dim=-1)
            nll = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), trues.view(-1), reduction='sum').item()
            total_nll += nll

    # Final metrics
    accuracy = total_correct / total_count
    perplexity = torch.exp(torch.tensor(total_nll / total_count))

    return {
        "accuracy": accuracy,
        "perplexity": perplexity.item()
    }
