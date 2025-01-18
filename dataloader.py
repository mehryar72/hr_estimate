import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, folder_path, index_list=None, full_len=0, seq_len=100, overlap=0.5):
        """
        Args:
            folder_path (str): Path to folder containing CSV files
            index_list (list): List of indices to include in dataset. If None, use all samples
            full_len (int): If 1, use full sequences. If 0, split into chunks
            seq_len (int): Length of sequences when full_len=0
            overlap (float): Overlap between sequences (0 to 1)
        """
        self.folder_path = folder_path
        self.full_len = full_len
        self.seq_len = seq_len
        
        # Get all CSV files in the folder
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if index_list is not None:
            self.file_list = [self.file_list[i] for i in index_list]
        
        # Pre-process all data at initialization
        self.samples = []
        
        # Process each file
        for file_name in self.file_list:
            file_path = os.path.join(folder_path, file_name)
            # Use memory mapping for large files
            df = pd.read_csv(file_path, memory_map=True)
            
            # Convert to numpy arrays once
            features = np.stack([df['ppg'].values, df['acc'].values], axis=1).astype(np.float32)
            labels = df['hr'].values.astype(np.float32)
            
            if full_len:
                # Store as tensors directly
                self.samples.append((
                    torch.from_numpy(features),
                    torch.from_numpy(labels)
                ))
            else:
                # Calculate step size based on overlap
                step = int(seq_len * (1 - overlap))
                
                # Calculate number of complete chunks
                num_chunks = (len(features) - 1) // step
                
                # Process all chunks including the last one
                for i in range(num_chunks + 1):
                    start_idx = i * step
                    end_idx = start_idx + seq_len
                    
                    # Handle the last chunk differently
                    if end_idx > len(features):
                        end_idx = len(features)
                        start_idx = max(0, end_idx - seq_len)  # Adjust start to maintain seq_len
                    
                    # Store as tensors directly
                    self.samples.append((
                        torch.from_numpy(features[start_idx:end_idx].copy()),
                        torch.from_numpy(labels[start_idx:end_idx].copy())
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # No need for conversion, already tensors
        return self.samples[idx]

def custom_collate_fn(batch):
    """
    Custom collate function using torch.nn.utils.rnn.pad_sequence.
    
    Args:
        batch: List of tuples (features, labels)
    
    Returns:
        padded_features: Tensor of padded features [batch_size, max_len, num_features]
        padded_labels: Tensor of padded labels [batch_size, max_len]
        lengths: List of original sequence lengths
        mask: Boolean mask tensor [batch_size, max_len] (True for actual values, False for padding)
    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Get lengths
    lengths = [len(seq) for seq in features]
    
    # Convert to tensors if they aren't already
    features = [torch.FloatTensor(feat) if not torch.is_tensor(feat) else feat for feat in features]
    labels = [torch.FloatTensor(lab) if not torch.is_tensor(lab) else lab for lab in labels]
    
    # Pad sequences
    padded_features = rnn_utils.pad_sequence(features, batch_first=True)
    padded_labels = rnn_utils.pad_sequence(labels, batch_first=True)
    
    # Create mask
    mask = torch.zeros(padded_labels.shape, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return padded_features, padded_labels, lengths, mask

def test_dataloader():
    # Create dataset
    dataset = TimeSeriesDataset(
        folder_path="./TrainData/",  # Replace with actual path
        full_len=1,
        # seq_len=100,
        # overlap=0.5
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # Test one iteration
    for features, labels, lengths, mask in dataloader:
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sequence lengths: {lengths}")
        print(f"Mask shape: {mask.shape}")
        print(f"Sample mask values:\n{mask}")
        break  # Only one iteration

if __name__ == "__main__":
    test_dataloader()