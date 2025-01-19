import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers.Transformer_EncDec import ConvLayer, Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer, FullAttention, ProbAttention
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
import torch.nn.functional as F
from layers.Conv_Blocks import Inception_Block_V1

class BiLSTMRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv= Inception(input_size=2, filters=config.hidCNN)
        self.lstm = nn.GRU(
            input_size=config.hidCNN*4,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        self.skip1 = nn.Sequential(nn.Conv1d(config.hidCNN*4, config.hidSkip, kernel_size=1, padding=0),nn.ReLU(),nn.BatchNorm1d(config.hidSkip))
        self.skip2 = nn.Sequential(nn.Conv1d(2, config.hidSkip, kernel_size=1, padding=0),nn.ReLU(),nn.BatchNorm1d(config.hidSkip))
        self.linear = nn.Linear(config.hidden_size * 2 + config.hidSkip * 2, 1, bias=False)  # *2 for bidirectional
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, x):
        # Check if input is packed
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        
        # Pass through LSTM
        if is_packed:
            x_unpac, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x_unpac=x_unpac.permute(0,2,1)
            x=self.conv(x_unpac)
            x_k=self.skip2(x_unpac).permute(0,2,1)
            x_k2 = self.skip1(x).permute(0,2,1)
            x=x.permute(0,2,1)
            x,_ = nn.utils.rnn.pack_padded_sequence(x, x.batch_sizes, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(x)
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            x_p=x.permute(0,2,1)
            x=self.conv(x_p)
            x_k2 = self.skip1(x).permute(0,2,1)
            x_k=self.skip2(x_p).permute(0,2,1)
            x=x.permute(0,2,1)
            output, _ = self.lstm(x)
        
        # Pass through linear layer
        predictions = self.linear(self.dropout(torch.cat((output, x_k,x_k2), 2)))
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



class deStationaryFromer (nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(deStationaryFromer, self).__init__()
        self.pred_len = configs.seq_len
        self.seq_len = configs.seq_len
        self.label_len = configs.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(c_in=2, d_model=configs.hidden_size,
                                           dropout=configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, 1, attention_dropout=configs.dropout,
                                    output_attention=False), configs.hidden_size, 8),
                    configs.hidden_size,
                    configs.hidden_size*4,
                    dropout=configs.dropout,
                    activation="gelu"
                ) for l in range(configs.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.hidden_size)
        )
        

        self.act = F.gelu
        self.dropout = nn.Sequential(nn.Dropout(configs.dropout), nn.Linear(configs.hidden_size, 1))
        self.tau_learner = Projector(enc_in=2, seq_len=configs.seq_len, hidden_dims=[128,128],
                                     hidden_layers=2, output_dim=1)
        self.delta_learner = Projector(enc_in=2, seq_len=configs.seq_len,
                                       hidden_dims=[128,128], hidden_layers=2,
                                       output_dim=configs.seq_len)

    

    def classification(self, x_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        return output

    def forward(self, x_enc):
        # x_enc = x_enc.permute(0, 2, 1)  # B x E x S
        dec_out = self.classification(x_enc)
        return dec_out.squeeze(-1)  # [B, L, D]

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y
    

class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Informer, self).__init__()
        self.pred_len = configs.seq_len
        self.label_len = configs.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(c_in=2, d_model=configs.hidden_size,embed_type="timeF",
                                           dropout=configs.dropout)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 1, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.hidden_size, 8),
                    configs.hidden_size,
                    configs.hidden_size * 4,
                    dropout=configs.dropout,
                    activation="gelu"
                ) for l in range(configs.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.hidden_size)
        )
        
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.hidden_size ,1)

    

    def forward(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)

        output = self.projection(output)  # (batch_size, num_classes)
        return output.squeeze(-1)


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.k = 5
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.hidden_size, configs.hidden_size*4,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(configs.hidden_size*4, configs.hidden_size,
                               num_kernels=6)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimeNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimeNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len

        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.num_layers)])
        self.enc_embedding = DataEmbedding(c_in=2, d_model=configs.hidden_size,embed_type="timeF",
                                           dropout=configs.dropout)
        self.layer = configs.num_layers
        self.layer_norm = nn.LayerNorm(configs.hidden_size)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.hidden_size ,1)
    def forward(self, x_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        output = self.projection(output)  # (batch_size, num_classes)
        return output.squeeze(-1)
