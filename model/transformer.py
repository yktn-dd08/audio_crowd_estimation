import torch
import torch.nn as nn
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification


class ASTRegressor(nn.Module):
    def __init__(self, feat_num=256, drop_out=0.3, out_features=1,
                 model_name='MIT/ast-finetuned-audioset-10-10-0.4593', finetune=False):
        super(ASTRegressor, self).__init__()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.ast_model = ASTForAudioClassification.from_pretrained(model_name)
        if not finetune:
            for param in self.ast_model.base_model.parameters():
                param.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Linear(in_features=self.ast_model.config.hidden_size, out_features=feat_num),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(in_features=feat_num, out_features=out_features)
        )

    def forward(self, x):
        x = self.ast_model(x, output_hidden_states=True)
        x = x.hidden_states[-1][:, 0, :]
        x = self.regressor(x)
        return x


class PositionalEncoding(nn.Module):
    """
    References: https://github.com/urbanaudiosensing/Models/blob/main/models/model.py
    """
    def __init__(self, segment_length, token_dim):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Embedding(segment_length, token_dim)
        self.pe_input = torch.Tensor(range(segment_length)).long()

    def forward(self, x):
        x = x + self.pe(self.pe_input.to(x.device))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元
        self.pe = pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class Conv1dTransformer(nn.Module):
    """
    1D-CNN + Transformer model
    x: Tensor [batch_size x freq_num x frame_num]
    """
    def __init__(self, freq_num, frame_num, kernel_size, dilation_size, pool_size,
                 token_dim=128, n_head=4, dff_times=4, drop_out=0.1, layer_num=1, pe_flag=True,
                 feat_num=128, out_features=1):
        super(Conv1dTransformer, self).__init__()
        # 1D-CNN
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=freq_num, out_channels=token_dim, kernel_size=kernel_size, dilation=dilation_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )
        # Positional Encoding
        self.pe_flag = pe_flag
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=token_dim, max_len=frame_num)

        # Conv1d適用後のフレーム数
        self.conv_time = int((frame_num - dilation_size * (kernel_size - 1)) / pool_size)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_head, dim_feedforward=token_dim*dff_times, dropout=drop_out, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=layer_num)

        # Regression
        self.regressor = nn.Sequential(
            nn.Linear(in_features=token_dim, out_features=feat_num),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(in_features=feat_num, out_features=out_features)
        )

    def forward(self, x):
        # x: batch_size x freq_num x frame_num
        x = self.conv(x)  # x: batch_size x token_dim x conv_time
        x = x.transpose(1, 2)  # x: batch_size x conv_time x token_dim
        if self.pe_flag:
            x = self.pos_encoding(x)
        x = self.transformer(x)  # x: batch_size x conv_time x token_dim
        x = x.mean(dim=1)  # x: batch_size x token_dim
        x = self.regressor(x)
        return x
