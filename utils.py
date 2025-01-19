import numpy as np
import torch
import os
import csv
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
def initialize_csv(log_file_path,cont=False):
    """Initialize the CSV file if it does not exist or delete it if it does."""
    if os.path.isfile(log_file_path) and not cont:
        os.remove(log_file_path)  # Delete the existing file
        # Create a new CSV file and write the header
        with open(log_file_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Accs'])  # Header

def log_training_values(log_file_path, epoch, train_loss, val_loss, avg_accuracy):
    """Log the training values to the CSV file."""
    with open(log_file_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        # Prepare the row to log
        log_row = [epoch, train_loss, val_loss] + list(avg_accuracy.values())
        log_writer.writerow(log_row)
