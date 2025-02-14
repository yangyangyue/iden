import copy
from typing import Optional
from torch.nn import functional as F
from torch import nn, Tensor
import math


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout: float = 0.0):
    """
    do attention operate
    Parameters:
        q: shape `(*, T, E)`
        k: shape `(*, S, E)`
        v: shape `(*, S, E)`
        mask: shape `(*, T, S)` ignore the attention weight where mask equals to 0
        dropout: ...
    T denote target sequence length, S denote source sequence length,
    E denote the embedding dimension also known as d_model/num_heads
    """
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout)

    return p_attn @ v, p_attn


class MultiheadAttention(nn.Module):
    """
    implement MultiHeadAttention mechanism
    Parameters:
        d_model:
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # the qkv is divided into num_heads parts,so increase 1D
            mask = mask.unsqueeze(1)
        bs = q.size(0)
        q, k, v = [l(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]

        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

        return self.linears[-1](x), attn


class Transformer(nn.Module):
    """
    hole transformer net, consist of encoder and decoder
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", norm_first=False,
                 custom_encoder: Optional[nn.Module] = None, custom_decoder: Optional[nn.Module] = None):
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                    norm_first)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                state_mask: Optional[Tensor] = None):
        state = self.encoder(src, src_mask=src_mask)
        out = self.decoder(tgt, state, tgt_mask=tgt_mask, state_mask=state_mask)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, state: Tensor,
                tgt_mask: Optional[Tensor] = None,
                state_mask: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, state, tgt_mask=tgt_mask, state_mask=state_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, mask: Optional[Tensor] = None):
        x = self.self_attn(x, x, x, mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm_first=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first

    def forward(self, tgt: Tensor, state: Tensor, tgt_mask: Optional[Tensor] = None,
                state_mask: Optional[Tensor] = None):

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._ca_block(self.norm2(x), state, state_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._ca_block(x, state, state_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, mask: Optional[Tensor] = None):
        x = self.self_attn(x, x, x, mask)[0]
        return self.dropout1(x)

    # cross-attention block
    def _ca_block(self, x: Tensor, mem: Tensor, mask: Optional[Tensor] = None):
        x = self.cross_attn(x, mem, mem, mask)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function from given string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
