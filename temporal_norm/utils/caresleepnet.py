import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from temporal_norm.utils._psdnorm import PSDNorm


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class MergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch * n_windows, n_chans, n_times)
        n_batch_times_n_windows, n_chans, n_times = x.shape
        n_batch = n_batch_times_n_windows // self.n_windows
        x = x.view(n_batch, self.n_windows, n_chans, n_times)
        x = x.permute(0, 2, 3, 1)  # (n_batch, n_chans, n_times, n_windows)
        x = x.reshape(n_batch, n_chans, n_times * self.n_windows)
        return x


class UnmergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch, n_chans, n_times * n_windows)
        n_batch, n_chans, total_time = x.shape
        n_times = total_time // self.n_windows
        x = x.view(n_batch, n_chans, n_times, self.n_windows)
        x = x.permute(0, 3, 1, 2)  # (n_batch, n_windows, n_chans, n_times)
        x = x.reshape(n_batch * self.n_windows, n_chans, n_times)
        return x


class MSDconv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        groups,
        n_windows=None,
        filter_size=None,
    ):
        super(MSDconv, self).__init__()
        self.downsample = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            groups=groups,
        )
        if filter_size is None:
            self.norm0 = nn.BatchNorm1d(out_planes)
        elif filter_size == 1:
            self.norm0 = nn.InstanceNorm1d(out_planes)
        elif filter_size > 1:
            self.norm0 = PSDNorm(
                filter_size=filter_size,
                n_channels=out_planes,
            )
        self.dconv1 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=1,
            groups=groups,
        )
        if filter_size is None:
            self.norm1 = nn.BatchNorm1d(out_planes)
        elif filter_size == 1:
            self.norm1 = nn.InstanceNorm1d(out_planes)
        elif filter_size > 1:
            self.norm1 = PSDNorm(
                filter_size=filter_size,
                n_channels=out_planes,
            )
        self.dconv2 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding * 2,
            bias=False,
            dilation=2,
            groups=groups,
        )
        if filter_size is None:
            self.norm2 = nn.BatchNorm1d(out_planes)
        elif filter_size == 1:
            self.norm2 = nn.InstanceNorm1d(out_planes)
        elif filter_size > 1:
            self.norm2 = PSDNorm(
                filter_size=filter_size,
                n_channels=out_planes,
            )
        self.dconv3 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding * 4,
            bias=False,
            dilation=4,
            groups=groups,
        )
        if filter_size is None:
            self.norm3 = nn.BatchNorm1d(out_planes)
        elif filter_size == 1:
            self.norm3 = nn.InstanceNorm1d(out_planes)
        elif filter_size > 1:
            self.norm3 = PSDNorm(
                filter_size=filter_size,
                n_channels=out_planes,
            )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(out_planes, eps=1e-6)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        down = self.norm0(self.downsample(x))
        x1 = F.gelu(self.norm1(self.dconv1(x)))
        x2 = F.gelu(self.norm2(self.dconv2(x)))
        x3 = F.gelu(self.norm3(self.dconv3(x)))
        out = down + x1 + x2 + x3
        out = self.dropout(out)
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

        return out


class UniConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        # vocab_size_or_config_json_file,
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        # self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style
        (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_probs


class UniEncoder(nn.Module):
    def __init__(self, config):
        super(UniEncoder, self).__init__()
        self.config = config
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        self.attention_probs = None

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers, attention_probs = self.encoder(
            x,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        self.attention_probs = attention_probs
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers[-1]


class EpochEncoder(nn.Module):
    def __init__(self, in_plane, n_windows=None, filter_size=None):
        super(EpochEncoder, self).__init__()
        self.encoder = nn.Sequential(
            MSDconv(
                in_plane,
                64,
                kernel_size=49,
                stride=12,
                padding=24,
                groups=1,
                filter_size=filter_size,
                n_windows=n_windows,
            ),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            MSDconv(64, 128, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(
                UniConfig(
                    hidden_size=128,
                    intermediate_size=512,
                )
            ),
            Transpose(),
            MSDconv(128, 256, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(UniConfig(hidden_size=256, intermediate_size=1024)),
            Transpose(),
            MSDconv(256, 512, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(UniConfig(hidden_size=512, intermediate_size=2048)),
            Transpose(),
        )

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        x = self.avg(x).squeeze()
        return x


class CareSleepNet(nn.Module):
    def __init__(
        self,
        n_chans,
        n_windows,
        dropout=0.1,
        n_outputs=5,
        filter_size=None,
    ):
        super(CareSleepNet, self).__init__()
        self.n_chans = n_chans
        self.n_windows = n_windows
        self.epoch_encoder_eeg = EpochEncoder(
            n_chans // 2, filter_size=filter_size, n_windows=n_windows
        )
        self.epoch_encoder_eog = EpochEncoder(
            n_chans // 2, filter_size=filter_size, n_windows=n_windows
        )
        self.eog2eeg_encoder = CMTransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.eeg2eog_encoder = CMTransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.sequence_encoder = TransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        x_eeg = x[:, :, : self.n_chans // 2, :]
        x_eeg = x_eeg.view(batch_size * self.n_windows, self.n_chans // 2, -1)
        x_eeg = self.epoch_encoder_eeg(x_eeg)
        x_eeg = x_eeg.view(batch_size, self.n_windows, -1)
        x_eog = x[:, :, self.n_chans // 2:, :]
        x_eog = x_eog.view(batch_size * self.n_windows, self.n_chans // 2, -1)
        x_eog = self.epoch_encoder_eog(x_eog)
        x_eog = x_eog.view(batch_size, self.n_windows, -1)
        x_eeg_ = self.eog2eeg_encoder(x_eeg, x_eog)
        x_eog_ = self.eeg2eog_encoder(x_eog, x_eeg)
        x = x_eeg_ + x_eog_

        x = self.sequence_encoder(x)

        return self.classifier(x).transpose(1, 2)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class CMEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_1_ = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, clue):
        torch._assert(
            input.dim() == 3,
            f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        z = self.ln_1_(clue)
        x, _ = self.self_attention(query=z, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class CMTransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layers = CMEncoderBlock(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, clue):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        return self.ln(self.layers(self.dropout1(input), self.dropout2(clue)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style(epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
