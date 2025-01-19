import argparse

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

def get_parser():
    parser = argparse.ArgumentParser(description='Time Series HR Prediction')
    
    # Dataset arguments
    parser.add_argument('--data_path',default="./TrainData" , type=str,
                      help='Path to folder containing CSV files')
    parser.add_argument('--seq_len', type=int, default=500,
                      help='Length of sequence chunks')
    parser.add_argument('--overlap', type=float, default=0.5,
                      help='Overlap between sequences (0 to 1)')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--full_len', type=int, default=0,
                      help='Use full sequence length (1) or chunk sequences (0)')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of layers in the model')
    parser.add_argument('--dropout', type=float, default=0.01,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=400,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay for optimizer')
    parser.add_argument('--loss_type', type=str, default='MAE',
                      help='Loss function to use')

    parser.add_argument('--hidCNN', type=int, default=32, help='Number of filters in the CNN layers')
    parser.add_argument('--hidSkip', type=int, default=64, help='Hidden size for the skip RNN')
    parser.add_argument('--skip', type=int, default=1, help='enable Skip')
    parser.add_argument('--last', type=int, default=64, help='Hidden size for the last fully connected layer')
    parser.add_argument('--scheduler', type=str, choices=['none','onecycle', 'reduce_on_plateau', 'triangular2'], default='none', help='Learning rate scheduler to use')
    return parser

# Global config instance
config = None

def get_config():
    global config
    if config is None:
        parser = get_parser()
        args = parser.parse_args()
        config = Config.from_args(args)
    return config
