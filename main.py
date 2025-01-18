import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from config import get_config
from dataloader import TimeSeriesDataset, custom_collate_fn
import numpy as np
from models import *
from utils import calculate_accuracy_margins
import torch.optim as optim

def get_loss_function(loss_type='mse'):
    if loss_type.lower() == 'mse':
        return nn.MSELoss(reduction='none')
    elif loss_type.lower() == 'mae':
        return nn.L1Loss(reduction='none')
    elif loss_type.lower() == 'huber':
        return nn.HuberLoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def train_epoch(model, dataloader, optimizer, loss_fn, device, full_len=False):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        if full_len:
            features, labels, lengths, mask = batch
        else:
            features, labels = batch
            lengths = [features.size(1)] * features.size(0)  # All same length
            mask = torch.ones_like(labels, dtype=torch.bool)
        
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Pack sequence if using padded data
        if full_len:
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=True
            )
            predictions = model(packed_features)
        else:
            predictions = model(features)
        
        # Calculate loss only on non-padded elements
        loss = loss_fn(predictions, labels)
        masked_loss = (loss * mask).sum() / mask.sum()  # Manual reduction
        
        masked_loss.backward()
        optimizer.step()
        
        total_loss += masked_loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate(model, dataloader, loss_fn, device, full_len=False):
    model.eval()
    total_loss = 0
    num_batches = 0
    total_counts = {margin: 0 for margin in [2, 3, 5, 10, 15, 20]}  # Initialize counts for each margin
    total_samples = 0  # Initialize total samples counter

    with torch.no_grad():
        for batch in dataloader:
            if full_len:
                features, labels, lengths, mask = batch
            else:
                features, labels = batch
                lengths = [features.size(1)] * features.size(0)  # All same length
                mask = torch.ones_like(labels, dtype=torch.bool)

            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            # Pack sequence if using padded data
            if full_len:
                packed_features = nn.utils.rnn.pack_padded_sequence(
                    features, lengths, batch_first=True, enforce_sorted=True
                )
                predictions = model(packed_features)
            else:
                predictions = model(features)

            loss = loss_fn(predictions, labels)
            masked_loss = (loss * mask).sum() / mask.sum()

            total_loss += masked_loss.item()
            num_batches += 1
            
            # Calculate accuracy margins for the current batch, considering the mask
            batch_accuracy = calculate_accuracy_margins(predictions, labels, mask)
            for margin in total_counts.keys():
                total_counts[margin] += batch_accuracy[f"within_{margin}"]

            # Update total samples processed
            total_samples += mask.sum().item()  # Count valid samples based on the mask

    # Convert counts to percentages
    accuracy_results = {f"within_{margin}": (count / total_samples) * 100 for margin, count in total_counts.items()}

    return total_loss / num_batches, accuracy_results

def test(model, dataset, config, device):
    """Test function that always receives full sequences and splits if needed."""
    model.eval()
    # all_predictions = []
    # all_targets = []
    total_counts = {margin: 0 for margin in [2, 3, 5, 10, 15, 20]}  # Initialize counts for each margin
    total_samples = 0  # Initialize total samples counter
    with torch.no_grad():
        if config.full_len:
            # Use the whole sequence
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                # collate_fn=custom_collate_fn
            )
            
            for features, labels, lengths, mask in dataloader:
                features = features.to(device)
                predictions = model(features)
                # all_predictions.extend(predictions[mask].cpu().numpy())
                # all_targets.extend(labels[mask].cpu().numpy())
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
                # all_predictions.extend(final_predictions)
                # all_targets.extend(labels)
    accuracy_results = {f"within_{margin}": (count / total_samples) * 100 for margin, count in total_counts.items()}
    return accuracy_results

def create_dataset_splits(data_path, train_ratio=0.8, val_ratio=0.15, test_ratio=0.0, seed=42, config=None):
    """Create train, validation, and test datasets using index lists. Test set always uses full sequences."""
    
    # Get total number of files
    file_list = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    total_files = len(file_list)
    
    # Calculate split sizes
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size
    
    # Create random indices
    np.random.seed(seed)
    indices = np.random.permutation(total_files)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Total files: {total_files}")
    print(f"Train files: {len(train_indices)}")
    print(f"Val files: {len(val_indices)}")
    print(f"Test files: {len(test_indices)}")
    
    # Create datasets - test set always uses full_len=1
    train_dataset = TimeSeriesDataset(
        folder_path=data_path,
        index_list=train_indices,
        full_len=config.full_len,
        seq_len=config.seq_len,
        overlap=config.overlap
    )
    
    val_dataset = TimeSeriesDataset(
        folder_path=data_path,
        index_list=val_indices,
        full_len=config.full_len,
        seq_len=config.seq_len,
        overlap=config.overlap
    )
    
    test_dataset = TimeSeriesDataset(
        folder_path=data_path,
        index_list=val_indices,
        full_len=1,  # Always use full sequences for test set
        seq_len=config.seq_len,
        overlap=config.overlap
    )
    
    return train_dataset, val_dataset, test_dataset

def get_scheduler(optimizer, scheduler_type, epochs, max_lr=0.01):
    if scheduler_type == 'none':
        return None  # No scheduler
    elif scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs)
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    elif scheduler_type == 'triangular2':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=40, mode='triangular2')
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

def main(base="LSTM"):
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets using index lists
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        data_path=config.data_path,
        train_ratio=0.85,
        val_ratio=0.15,
        # test_ratio=0.5,
        config=config
    )
    
    # Create dataloaders - use collate_fn when  full_len
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn if config.full_len else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn if config.full_len else None
    )
    
    # Create model and move to device
    # model = BiLSTMRegressor(
    #     hidden_size=config.hidden_size,
    #     num_layers=config.num_layers,
    #     dropout=config.dropout
    # ).to(device)
    model = Model3CNNRNN(config, base=base).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    loss_fn = get_loss_function(config.loss_type)
    
    # Get the learning rate scheduler
    scheduler = get_scheduler(optimizer, config.scheduler, config.epochs,config.learning_rate)
    
    # Training loop
    print("\nStarting training...")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Val Accs':>20} {'Best':>6}")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_accs = None  # To store best validation accuracies
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config.full_len)
        val_loss, accuracy_results = validate(model, val_loader, loss_fn, device, config.full_len)

        # Calculate average accuracy for printing
        avg_accuracy = {margin: f"{accuracy_results.get(f'within_{margin}', 0):.2f}%" for margin in [2, 3, 5, 10, 15, 20]}
        
        # Print validation loss and accuracy results
        print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} {' | '.join(avg_accuracy.values()):>20} {'*' if val_loss < best_val_loss else ''}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_val_accs = accuracy_results  # Save the best validation accuracies
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accs': best_val_accs  # Save validation accuracies
            }, 'best_model.pth')
        
        # Step the scheduler if using ReduceLROnPlateau
        if config.scheduler == 'none':
            pass
        elif config.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)  # Assuming val_loss is available
        
        # For other schedulers, step the scheduler at the end of each epoch
        else:
            scheduler.step()

    print("-" * 70)
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Best validation accuracies: {best_val_accs}")

    # Testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    accuracy_results = test(model, test_dataset, config, device)
    
    print("\nTest Results:")
    print("Accuracy Results:")
    for margin, percentage in accuracy_results.items():
        margin_value = margin.split('_')[1]  # Extract number from 'within_X'
        print(f"±{margin_value}: {percentage:.2f}%")
    # Calculate final metrics
    # mse = np.mean((predictions - targets) ** 2)
    # mae = np.mean(np.abs(predictions - targets))
    # rmse = np.sqrt(mse)
    
    # print("\nTest Results:")
    # print(f"MSE:  {mse:.6f}")
    # print(f"RMSE: {rmse:.6f}")
    # print(f"MAE:  {mae:.6f}")
    
    # Calculate and print accuracy margins
    # accuracy_results = calculate_accuracy_margins(predictions, targets)
    # print_accuracy_results(accuracy_results)

if __name__ == "__main__":
    main(base="LSTM")
