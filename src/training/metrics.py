import torch
import torch.nn.functional as F

def compute_metrics(predictions, targets):
    # Stack all predictions/targets into a single tensor
    preds = torch.cat(predictions, dim=0)
    trues = torch.cat(targets, dim=0)

    # Get top predictions
    pred_ids = preds.argmax(dim=-1)

    # Accuracy
    correct = (pred_ids == trues).float()
    accuracy = correct.sum() / correct.numel()

    # Perplexity
    with torch.no_grad():
        log_probs = F.log_softmax(preds, dim=-1)
        nll = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), trues.view(-1), reduction='mean')
        perplexity = torch.exp(nll)

    return {
        "accuracy": accuracy.item(),
        "perplexity": perplexity.item()
    }
