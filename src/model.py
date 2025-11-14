import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Encoder(nn.Module):
    """An encoder model with ACF_attention mechanism."""

    def __init__(self, n_layers, dimension, d_hid,num_heads, d_inner, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dimension, d_hid, num_heads,d_inner, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm([dimension], eps=1e-6)

    def forward(self, x_batch, save_attention=False):
        enc_output = self.dropout(x_batch)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, save_attention)

        return enc_output


class Decoder(nn.Module):
    """A decoder model with ACF_attention mechanism."""

    def __init__(self, n_layers, dimension, d_hid,num_heads, d_inner, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(dimension, d_hid, num_heads,d_inner, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm([d_hid], eps=1e-6)

    def forward(self, x_batch, enc_output):
        dec_output = self.dropout(x_batch)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)

        return dec_output


class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()

        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):

        batch_size = x.size(0)

        query = self.query_transform(x)  # [seq_len, batch_size, feature_dim]
        keys = self.key_transform(x)     # [seq_len, batch_size, feature_dim]

        values = self.value_transform(x) # [seq_len, batch_size, feature_dim]

        d_k = query.size(-1)
        scores = torch.bmm(query.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2)) / math.sqrt(d_k)  # [batch_size, seq_len, seq_len]

        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]

        attention_output = torch.bmm(attention_weights, values.transpose(0, 1)).sum(dim=-1)  # [batch_size, seq_len, feature_dim]
        attention_output = attention_output


        return attention_output


class MultiScaleSTFTEncoder(nn.Module):
    def __init__(self, base_scales, base_window_sizes, base_hop_sizes, input_length, num_features):
        super(MultiScaleSTFTEncoder, self).__init__()
        self.base_scales = base_scales
        self.base_window_sizes = base_window_sizes
        self.base_hop_sizes = base_hop_sizes
        self.scale_factor = nn.Parameter(torch.ones(len(base_scales)))
        self.hop_factor = nn.Parameter(torch.ones(len(base_hop_sizes)))
        self.input_length = input_length
        self.num_features = num_features
        self.attention = SimpleAttention(len(base_scales))
        self.layer_norm = nn.LayerNorm([num_features], eps=1e-6)
        self.query_transform = nn.Linear(len(base_scales), len(base_scales))
        self.key_transform = nn.Linear(len(base_scales), len(base_scales))
        self.value_transform = nn.Linear(len(base_scales), len(base_scales))
        self.feature_transforms = nn.ModuleList([
            nn.Linear((base_hop_sizes[i]//2 + 1) * (input_length // base_hop_sizes[i] + 1),
                      (base_hop_sizes[i]//2 + 1) * (input_length // base_hop_sizes[i] + 1))
            for i in range(len(base_hop_sizes))
        ])


    def forward(self, x, save_attention=False):
        batch_size, length, num_features = x.shape
        scale_results = []

        for feature_index in range(num_features):
            feature_scale_results = []
            for i, scale in enumerate(self.base_scales):
                adjusted_window_size = max(int((self.base_window_sizes[i] * self.scale_factor[i]).detach()), 1)


                adjusted_hop_size = max(
                    min(int((self.base_hop_sizes[i] * self.hop_factor[i]).detach()), adjusted_window_size),
                    1
                )

                scale_x = x[:, :, feature_index]

                stft_result = torch.stft(scale_x, n_fft=adjusted_window_size, hop_length=adjusted_hop_size,
                                         window=torch.ones(adjusted_window_size).to(x.device),
                                         return_complex=True)

                magnitude = torch.abs(stft_result)
                phase = torch.angle(stft_result)


                energy_per_frame = magnitude.pow(2).sum(dim=1)


                importance_per_frame = F.softmax(energy_per_frame, dim=-1)


                importance_per_frame = importance_per_frame.unsqueeze(1).expand_as(magnitude)
                weighted_magnitude = magnitude * importance_per_frame



                feature_maps = weighted_magnitude.reshape(batch_size, -1)

                transformed_features = self.feature_transforms[i](feature_maps)

                weighted_features = (torch.relu(transformed_features) * transformed_features ).view_as(magnitude)

                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                real_part = weighted_features * cos_phase
                imag_part = weighted_features * sin_phase
                adjusted_stft = torch.complex(real_part, imag_part)
                reconstructed_signal = torch.istft(adjusted_stft, n_fft=adjusted_window_size,
                                                   hop_length=adjusted_hop_size,
                                                   window=torch.ones(adjusted_window_size).to(x.device))

                feature_scale_results.append(reconstructed_signal)


            scale_results.append(torch.stack(feature_scale_results))

        results = torch.stack(scale_results).view(batch_size, length, num_features, -1)
        # print("results: ", results.shape)
        results1 = self.query_transform(results)
        results2 = self.key_transform(results)
        results3 = self.value_transform(results)

        attention_scores = torch.matmul(results1,
                                        results2.transpose(-2, -1))  # [batch_size, length, num_features, num_features]

        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, length, num_features, num_features]


        attention_output = torch.matmul(attention_weights, results3)  # [batch_size, length, num_features, dim]


        results5 = self.layer_norm(attention_output.sum(dim=-1))

        if save_attention:
            self.multi_map = results.detach().cpu().numpy()

        return results5


class Transformer(nn.Module):
    """带有增强多尺度注意力机制的序列到序列模型。"""
    def __init__(self, dimension=6, num_heads=1, d_hid=250, d_inner=250, n_layers=3, dropout=0,mul_dim=16, step_size=10, sample_length=400):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_hid, eps=1e-6)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_hid, dimension))
        self.encoder = Encoder(n_layers=n_layers, dimension=dimension, d_hid=d_hid, num_heads=num_heads, d_inner=d_inner, dropout=dropout)
        self.decoder = Decoder(n_layers=n_layers, dimension=dimension, d_hid=d_hid, num_heads=num_heads, d_inner=d_inner, dropout=dropout)
        self.mlp_layers = nn.Sequential(nn.Linear(d_hid, d_inner))
        self.mlp_layers1 = nn.Sequential(nn.Linear(dimension, 1))


    def forward(self, src_seq, save_attention=False):

        enc_output = self.encoder(x_batch=src_seq, save_attention=save_attention)
        enc_output = enc_output.mean(dim=-1)

        seq_logit = self.mlp_layers(enc_output)
        return seq_logit


class EncoderLayer(nn.Module):
    """Composed of two sub-layers: self-attention and position-wise feed-forward"""

    def __init__(self, dimension, d_hid, num_heads,d_inner, dropout):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(dimension=dimension, num_heads=num_heads, dropout=dropout)
        self.multi = MultiScaleSTFTEncoder(base_scales=[40,100,200],  base_window_sizes=[40,100,200],
                                            base_hop_sizes=[40,100,200],
                                        input_length=400,num_features=6)
        self.pos_ffn = PositionwiseFeedForward(d_hid=d_hid, d_inner=d_inner, dropout=dropout)

    def forward(self, enc_input, save_attention):
        enc_slf_attn = self.slf_attn(enc_input, save_attention)
        enc_mulit = self.multi(enc_input, save_attention)
        enc_slf_attn = enc_slf_attn + enc_mulit
        enc_output = self.pos_ffn(enc_slf_attn)
        return enc_output


class DecoderLayer(nn.Module):
    """Composed of three sub-layers: self-attention, encoder-decoder attention, and position-wise feed-forward"""

    def __init__(self, dimension, d_hid,num_heads, d_inner, dropout):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(dimension=dimension, num_heads=num_heads, dropout=dropout)
        self.enc_attn = MultiHeadAttention(dimension=dimension, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm([d_hid], eps=1e-6)
        self.pos_ffn = PositionwiseFeedForward(d_hid=d_hid, d_inner=d_inner, dropout=dropout)

    def forward(self, dec_input):
        dec_slf_attn = self.slf_attn(dec_input)
        dec_enc_attn = self.enc_attn(dec_slf_attn)
        dec_output = self.pos_ffn(dec_enc_attn)
        return dec_output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with autocorrelation features."""

    def __init__(self, dimension, num_heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.acf_attention = AutocorrModule(dimension, num_heads)
        self.attention_map = None

    def forward(self, x, save_attention=False):
        autocorr_matrix, att_score = self.acf_attention(x)
        if save_attention:
            self.attention_map = att_score.squeeze().detach().cpu().numpy()
        attn_output = self.layer_norm(autocorr_matrix + x)

        return attn_output


class AutocorrModule(nn.Module):
    def __init__(self, dimension, num_heads):
        super(AutocorrModule, self).__init__()
        self.num_heads = num_heads
        self.dimension = dimension
        self.head_dim = dimension // num_heads
        self.layer_norm1 = nn.LayerNorm(dimension, eps=1e-6)

        assert dimension % num_heads == 0, "d_model must be divisible by num_heads"

        self.layer_norm = nn.LayerNorm([1], eps=1e-6)  # 根据实际输入大小调整
        self.query_weights = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.key_weights = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.value_weights = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])

        self.query_weight1 = nn.Linear(dimension, dimension)
        self.key_weight1 = nn.Linear(dimension, dimension)
        self.value_weight1 = nn.Linear(dimension, dimension)

    def forward(self, x):
        batch_size, seq_length, feature = x.size()


        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)


        Q, K, V = [], [], []
        for i in range(self.num_heads):
            Q.append(self.query_weights[i](x[:, i, :, :]))
            rolled_right = torch.roll(x[:, i, :, :], shifts=1, dims=1)
            rolled_right[:, 0] = x[:, i, 0]  # 将最左边的值用来填充滑动后新出现的空缺位置
            K.append(self.key_weights[i](rolled_right))

            rolled_left = torch.roll(x[:, i, :, :], shifts=-1, dims=1)
            rolled_left[:, -1] = x[:, i, -1]  # 将最右边的值用来填充滑动后新出现的空缺位置
            V.append(self.value_weights[i](rolled_left))

        multi_head_output = []
        multi_teo = []
        for q, k, v in zip(Q, K, V):
            teo = q ** 2 - k * v
            # print("teo",teo.size())
            attention_scores = torch.softmax(self.layer_norm(teo),dim=1)
            weighted_features = attention_scores * teo
            multi_head_output.append(weighted_features)
            multi_teo.append(teo)
        multi_teos = torch.stack(multi_teo, dim=-1).squeeze()

        multi_head_output1 = torch.cat(multi_head_output, dim=-1).squeeze()


        output = multi_head_output1


        return output, multi_teos


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_hid, d_inner, dropout):
        super().__init__()
        self.w_1 = nn.Linear(6, 64)  # Position-wise linear layer 1
        self.w_2 = nn.Linear(64, 6)  # Position-wise linear layer 2
        self.layer_norm = nn.LayerNorm(6, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(self.w_2(x))
        x += residual
        x = self.layer_norm(x)
        return x
