import torch
import numpy as np
from sklearn.metrics import accuracy_score


def calculate_perplexity(model, data_loader, vocab):
    """
    Calculate perplexity of a model on a dataset.
    Returns both overall and per-sentence perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            
            # Mask out padding tokens
            mask = (targets != vocab['<PAD>']).float()
            num_tokens = mask.sum().item()
            
            # Calculate loss only on non-padding tokens
            loss = model.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = loss.view(targets.size()) * mask
            
            total_loss += loss.sum().item()
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_accuracy(model, data_loader, vocab):
    """
    Calculate prediction accuracy of a model on a dataset.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            
            # Get predictions (highest probability word)
            preds = outputs.argmax(dim=-1)
            
            # Only consider non-padding positions
            mask = targets != vocab['<PAD>']
            valid_preds = preds[mask].cpu().numpy()
            valid_targets = targets[mask].cpu().numpy()
            
            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    return accuracy * 100  # Convert to percentage
