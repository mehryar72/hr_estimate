import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(2, 64, kernel_size=7, padding=3),nn.ReLU(),nn.BatchNorm1d(64))
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
    
    def forward(self, x):
        # Check if input is packed
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        
        # Pass through LSTM
        if is_packed:
            x_unpac, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x_unpac=x_unpac.permute(0,2,1)
            x=self.conv(x_unpac)
            x=x.permute(0,2,1)
            x,_ = nn.utils.rnn.pack_padded_sequence(x, x.batch_sizes, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(x)
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            x=x.permute(0,2,1)
            x=self.conv(x)
            x=x.permute(0,2,1)
            output, _ = self.lstm(x)
        
        # Pass through linear layer
        predictions = self.linear(output)
        return predictions.squeeze(-1)  # Remove last dimension

class Model2CNNRNN(nn.Module):
    def __init__(self, config, base="LSTM"):
        super(Model2CNNRNN, self).__init__()
        self.last = config.last
        self.hidR = config.hidden_size
        self.hidC = config.hidCNN
        self.hidS = config.hidSkip
        self.skip = config.skip
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv1d(2, self.hidC, kernel_size=3, padding=(3 - 1) // 2)
        self.bn1 = nn.BatchNorm1d(self.hidC)  # Batch Normalization after conv1
        self.conv2 = nn.Conv1d(self.hidC, self.hidC * 2, kernel_size=5, padding=(5 - 1) // 2)
        self.bn2 = nn.BatchNorm1d(self.hidC * 2)  # Batch Normalization after conv2
        self.conv3 = nn.Conv1d(self.hidC * 2, self.hidC * 2, kernel_size=5, padding=(5 - 1) // 2)
        self.bn3 = nn.BatchNorm1d(self.hidC * 2)  # Batch Normalization after conv3
        
        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.bn1,
            self.conv2,
            nn.ReLU(),
            self.bn2,
            self.conv3,
            nn.ReLU(),
            self.bn3
        )
        
        # RNN layers
        if base == "LSTM":
            self.RNN1 = nn.LSTM(self.hidC * 2, self.hidR, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
        else:
            self.RNN1 = nn.GRU(self.hidC * 2, self.hidR, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Skip RNN
        if self.skip > 0:
            if base == "LSTM":
                self.RNNskip = nn.LSTM(2, self.hidS, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
            else:
                self.RNNskip = nn.GRU(2, self.hidS, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
            self.linear1 = nn.Sequential(nn.Linear(self.hidR * 2 + self.skip * self.hidS * 2, self.last), nn.ReLU())
        else:
            self.linear1 = nn.Sequential(nn.Linear(self.hidR * 2, self.last), nn.ReLU())
        
        self.linear2 = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(self.last, 1))

    def forward(self, x):
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            x_unpac, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x_unpac = x
        
        x_unpac = x_unpac.permute(0, 2, 1)  # Change shape for Conv1d
        c = self.conv(x_unpac)  # Pass through conv layers
        c = self.dropout(c)
        r = c.permute(0, 2, 1)  # Change shape back for RNN
        
        if is_packed:
            c = nn.utils.rnn.pack_padded_sequence(c, x.batch_sizes, batch_first=True, enforce_sorted=False)
            r, _ = self.RNN1(c)
            r, _ = nn.utils.rnn.pad_packed_sequence(r, batch_first=True)
        else:
            r, _ = self.RNN1(r)
        
        r = self.dropout(r)
        
        if self.skip > 0:
            r2, _ = self.RNNskip(x)
            if is_packed:
                r2, _ = nn.utils.rnn.pad_packed_sequence(r2, batch_first=True)
            r2 = self.dropout(r2)
            r = torch.cat((r, r2), 2)
        
        res = self.linear1(r)
        res = self.linear2(res)

        return res.squeeze(-1)


class Model3CNNRNN(nn.Module):
    def __init__(self, config, base="LSTM"):
        super(Model3CNNRNN, self).__init__()
        self.last = config.last
        self.hidR = config.hidden_size
        self.hidC = config.hidCNN
        self.hidS = config.hidSkip
        self.skip = config.skip
        
        # Inception block with input size of 2 and number of filters as hidC
        self.inception = Inception(input_size=2, filters=self.hidC)
        
        # RNN layers
                # RNN layers
        if base == "LSTM":
            self.RNN1 = nn.LSTM(self.hidC * 4, self.hidR, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
        else:
            self.RNN1 = nn.GRU(self.hidC * 4, self.hidR, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Skip path with additional conv layer
        self.skip_conv = nn.Conv1d(2, 32, kernel_size=1)  # Expand from 2 to 32 channels
        self.skip_bn = nn.BatchNorm1d(32)  # Batch Normalization
        self.skip_activation = nn.ReLU()  # ReLU activation
        
        if self.skip > 0:
            if base == "LSTM":
                self.RNNskip = nn.LSTM(32, self.hidS, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
            else:
                self.RNNskip = nn.GRU(32, self.hidS, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout=config.dropout if config.num_layers > 1 else 0)
            self.linear1 = nn.Sequential(nn.Linear(self.hidR * 2 + self.skip * self.hidS * 2, self.last), nn.ReLU())
        else:
            self.linear1 = nn.Sequential(nn.Linear(self.hidR * 2, self.last), nn.ReLU())
        
        self.linear2 = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(self.last, 1))

    def forward(self, x):
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            x_unpac, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x_unpac = x
        
        # Pass through Inception block
        c = self.inception(x_unpac.permute(0, 2, 1))  # Change shape for Inception
        # c = self.dropout(c)
        
        r = c.permute(0, 2, 1)  # Change shape back for RNN
        if is_packed:
            c = nn.utils.rnn.pack_padded_sequence(c, x.batch_sizes, batch_first=True, enforce_sorted=False)
            r, _ = self.RNN1(c)
            r, _ = nn.utils.rnn.pad_packed_sequence(r, batch_first=True)
        else:
            r, _ = self.RNN1(r)  # Pass through RNN
        
        # r = self.dropout(r)
        
        if self.skip > 0:
            # Process through the skip path
            r2 = self.skip_conv(x_unpac.permute(0,2,1))  # Apply the 1x1 convolution
            r2 = self.skip_bn(r2)  # Apply Batch Normalization
            r2 = self.skip_activation(r2)  # Apply ReLU activation
            
            r2 = r2.permute(0, 2, 1)  # Change shape for RNN
            if is_packed:
                r2 = nn.utils.rnn.pack_padded_sequence(r2, x.batch_sizes, batch_first=True, enforce_sorted=False)
                r2, _ = self.RNNskip(r2)
                r2, _ = nn.utils.rnn.pad_packed_sequence(r2, batch_first=True)
            else:
                r2, _ = self.RNNskip(r2)  # Pass through skip RNN
            
            if is_packed:
                r2, _ = nn.utils.rnn.pad_packed_sequence(r2, batch_first=True)
            # r2 = self.dropout(r2)
            r = torch.cat((r, r2), 2)  # Concatenate RNN outputs
        
        res = self.linear1(r)
        res = self.linear2(res)

        return res.squeeze(-1)  # Remove last dimension


class Inception(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Inception, self).__init__()
        
        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv10 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv20 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv40 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.bottleneck2 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.bottleneck2(self.max_pool(x))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = torch.nn.functional.relu(self.batch_norm(y))
        return y


class Residual(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()
        
        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )
    
    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = torch.nn.functional.relu(y)
        return y


class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)


class InceptionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes, filters, depth):
        super(InceptionModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        
        modules = OrderedDict()
        
        for d in range(depth):
            modules[f'inception_{d}'] = Inception(
                input_size=input_size if d == 0 else 4 * filters,
                filters=filters,
            )
            if d % 3 == 2:
                modules[f'residual_{d}'] = Residual(
                    input_size=input_size if d == 2 else 4 * filters,
                    filters=filters,
                )
        
        modules['linear'] = torch.nn.Linear(in_features=4 * filters, out_features=num_classes)
        
        self.model = torch.nn.Sequential(modules)

    def forward(self, x):
        for d in range(self.depth):
            y = self.model.get_submodule(f'inception_{d}')(x if d == 0 else y)
            if d % 3 == 2:
                y = self.model.get_submodule(f'residual_{d}')(x, y)
                x = y

        y = self.model.get_submodule('linear')(y)
        return y

 # Remove last dimension