import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import get_config
from dataloader import TimeSeriesDataset  # Adjust the import based on your dataset class
from models import BiLSTMRegressor, Informer, Model3CNNRNN
from utils import calculate_accuracy_margins  # Adjust the import based on your model class

def load_best_model(config):
    """Load the best model from the checkpoint."""
    model_path = "./checkpoints/best_model.pth"
    checkpoint = torch.load(model_path)
    model_type = checkpoint.get("modeltype")
    if os.path.isfile(model_path):
        if model_type=="custom":
            model = Informer(config)
        else:
            model = Model3CNNRNN(config, base=model_type)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set the model to evaluation mode
        return model
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

def evaluate_model(model, dataset, config, device):
    """Evaluate the model on the test dataset and return accuracy results."""
    model.eval()
    total_counts = {margin: 0 for margin in [2, 3, 5, 10, 15, 20]}  # Initialize counts for each margin
    total_samples = 0  # Initialize total samples counter

    with torch.no_grad():
        if config.full_len:
            # Use the whole sequence
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
            )
            
            for features, labels, lengths, mask in dataloader:
                features = features.to(device)
                predictions = model(features)
                batch_accuracy = calculate_accuracy_margins(predictions, labels, mask)
                for margin in total_counts.keys():
                    total_counts[margin] += batch_accuracy[f"within_{margin}"]
                total_samples += mask.sum().item()
        else:
            # Manual splitting of sequences
            for features, labels in dataset.samples:
                features = torch.FloatTensor(features).unsqueeze(0).to(device)
                seq_len = features.size(1)
                
                # Split sequence into chunks
                chunk_predictions = []
                for i in range(0, seq_len - config.seq_len + 1, int(config.seq_len * (1 - config.overlap))):
                    end_idx = min(i + config.seq_len, seq_len)
                    chunk = features[:, i:end_idx, :]
                    chunk_len = [chunk.size(1)]
                    
                    pred = model(chunk)
                    chunk_predictions.append(pred.cpu().numpy())
                
                # Aggregate predictions (simple averaging for overlapping parts)
                final_predictions = np.zeros(seq_len)
                counts = np.zeros(seq_len)
                
                for i, preds in enumerate(chunk_predictions):
                    start_idx = i * int(config.seq_len * (1 - config.overlap))
                    end_idx = start_idx + len(preds[0])
                    final_predictions[start_idx:end_idx] += preds[0]
                    counts[start_idx:end_idx] += 1
                
                final_predictions = final_predictions / counts
                batch_accuracy = calculate_accuracy_margins(final_predictions, labels, torch.ones_like(labels, dtype=torch.bool))
                for margin in total_counts.keys():
                    total_counts[margin] += batch_accuracy[f"within_{margin}"]
                total_samples += labels.size(0)

    # Convert counts to percentages
    accuracy_results = {f"within_{margin}": (count / total_samples) * 100 for margin, count in total_counts.items()}
    return accuracy_results

def main():
    # Load configuration
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load normalization parameters from the checkpoint
    checkpoint_path = "./checkpoints/best_model.pth"
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        train_mean = checkpoint.get("train_mean")
        train_var = checkpoint.get("train_var")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Create test dataset
    test_dataset = TimeSeriesDataset(folder_path="./ValidData/", full_len=1)  # Assuming full_len=1 for testing

    # Normalize the test dataset using the loaded mean and variance
    test_dataset.normilise(train_mean, train_var)

    # Load the best model
    model = load_best_model(config).to(device)

    # Evaluate the model
    accuracy = evaluate_model(model, test_dataset, config, device)

    # Print the accuracy results
    print("Accuracy of the best model on the test dataset:")
    for margin, percentage in accuracy.items():
        margin_value = margin.split('_')[1]  # Extract number from 'within_X'
        print(f"±{margin_value}: {percentage:.2f}%")
    #save inference results as json
    with open('inference_results.json', 'w') as f:
        json.dump(accuracy, f)

if __name__ == "__main__":
    main()
