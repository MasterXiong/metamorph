import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList

from metamorph.config import cfg


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, context=None, morphology_info=None):
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for l in self.layers:
            output = l(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, context=context, morphology_info=morphology_info)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def get_attention_maps(self, src, mask=None, src_key_padding_mask=None, context=None, morphology_info=None):
        attention_maps = []
        output = src

        for l in self.layers:
            # NOTE: Shape of attention map: Batch Size x MAX_JOINTS x MAX_JOINTS
            # pytorch avgs the attention map over different heads; in case of
            # nheads > 1 code needs to change.
            output, attention_map = l(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True, 
                context=context, 
                morphology_info=morphology_info
            )
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps


class TransformerEncoderLayerResidual(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayerResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if cfg.MODEL.TRANSFORMER.FIX_ATTENTION:
            self.norm_context = nn.LayerNorm(d_model)
        
        if cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            self.connectivity_encoder = nn.Sequential(
                nn.Linear(3, 16), 
                nn.ReLU(), 
                nn.Linear(16, nhead), 
            )

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayerResidual, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False, context=None, morphology_info=None):
        
        src2 = self.norm1(src)

        if not cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            src_mask = None
        else:
            # (batch_size, seq_len, seq_len, feat_dim) -> (batch_size, seq_len, seq_len, num_head)
            src_mask = self.connectivity_encoder(morphology_info['connectivity'])
            # (batch_size, seq_len, seq_len, num_head) -> (batch_size * num_head, seq_len, seq_len)
            # stack the embedding for each head in the first dimension
            src_mask = torch.cat([src_mask[:, :, :, i] for i in range(src_mask.size(-1))], 0)

        if context is not None:
            context_normed = self.norm_context(context)
            src2, attn_weights = self.self_attn(
                context_normed, context_normed, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
        else:
            src2, attn_weights = self.self_attn(
                src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
            )
        
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, attn_weights
        else:
            return src
        