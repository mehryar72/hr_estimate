import numpy as np
import torch

def calculate_accuracy_margins(predictions, targets, mask):
    """
    Calculate number of predictions within different margins of error, considering a mask.
    
    Args:
        predictions: Model predictions for HR
        targets: True HR values
        mask: Boolean mask indicating valid entries
        
    Returns:
        dict: Number of predictions within each margin
    """
    # Convert to numpy if tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    # Apply mask to predictions and targets
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]

    # Calculate absolute differences
    abs_diff = np.abs(valid_predictions - valid_targets)
    
    # Define margins
    margins = [2, 3, 5, 10, 15, 20]
    total_samples = len(valid_predictions)
    
    # Calculate counts for each margin
    results = {}
    for margin in margins:
        count = np.sum(abs_diff <= margin)
        results[f"within_{margin}"] = count  # Store count directly
    
    return results

def print_accuracy_results(results):
    """Print accuracy results in a simplified way."""
    print("\nAccuracy Results:")
    for margin, percentage in results.items():
        margin_value = margin.split('_')[1]  # Extract number from 'within_X'
        print(f"±{margin_value}: {percentage:.2f}%")
