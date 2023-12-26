import numpy as np
import torch
import torch.nn as nn

from metamorph.config import cfg
from metamorph.utils import model as tu
from .transformer import TransformerEncoder, TransformerEncoderLayerResidual


class ContextEncoder(nn.Module):
    def __init__(self, obs_space):
        super(ContextEncoder, self).__init__()
        self.model_args = cfg.MODEL.HYPERNET
        self.seq_len = cfg.MODEL.MAX_LIMBS
        context_obs_size = obs_space["context"].shape[0] // self.seq_len

        if self.model_args.CONTEXT_ENCODER_TYPE == 'linear':
            context_encoder_dim = [context_obs_size] + [self.model_args.CONTEXT_EMBED_SIZE for _ in range(self.model_args.ENCODER_LAYER_NUM)]
            self.context_encoder = tu.make_mlp_default(context_encoder_dim)
        elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
            context_embed = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            context_encoder_layers = TransformerEncoderLayerResidual(
                self.model_args.CONTEXT_EMBED_SIZE,
                cfg.MODEL.TRANSFORMER.NHEAD,
                256,
                cfg.MODEL.TRANSFORMER.DROPOUT, 
                batch_first=True, 
            )
            context_encoder_TF = TransformerEncoder(
                context_encoder_layers, 1, norm=None,
            )
            if self.model_args.CONTEXT_MASK:
                self.context_embed = context_embed
                self.context_encoder = context_encoder_TF
            else:
                self.context_encoder = nn.Sequential(
                    context_embed, 
                    context_encoder_TF, 
                )
        # elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
        #     # self.context_embed_input = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
        #     self.context_encoder_for_input = GraphNeuralNetwork(
        #         input_dim = context_obs_size, 
        #         hidden_dims = [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE], 
        #         output_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE, 
        #         final_nonlinearity=True
        #     )
        if self.model_args.EMBEDDING_DROPOUT is not None:
            self.embedding_dropout = nn.Dropout(p=self.model_args.EMBEDDING_DROPOUT)

    def forward(self, obs_context, obs_mask):
        context_embedding = obs_context
        if self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
            if self.model_args.CONTEXT_MASK:
                context_embedding = self.context_embed(context_embedding)
                context_embedding = self.context_encoder(context_embedding, src_key_padding_mask=obs_mask)
            else:
                context_embedding = self.context_encoder(context_embedding)
        else:
            context_embedding = self.context_encoder(context_embedding)
        if self.model_args.EMBEDDING_DROPOUT is not None:
            context_embedding = self.embedding_dropout(context_embedding)
        return context_embedding


class HypernetLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim):
        super(HypernetLayer, self).__init__()
        self.base_input_dim = base_input_dim
        self.base_output_dim = base_output_dim
        self.model_args = cfg.MODEL.HYPERNET
        HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE

        self.HN_weight = nn.Linear(HN_input_dim, base_input_dim * base_output_dim)
        self.HN_bias = nn.Linear(HN_input_dim, base_output_dim)

        self.init_hypernet()

    def init_hypernet(self):
        if self.model_args.HN_INIT_STRATEGY == 'bias_init':
            initrange = np.sqrt(1 / self.base_input_dim)
            self.HN_weight.weight.data.zero_()
            self.HN_weight.bias.data.normal_(std=initrange)
        elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
            initrange = np.sqrt(1 / self.base_input_dim)
            self.HN_weight.weight.data.zero_()
            self.HN_weight.bias.data.uniform_(-initrange, initrange)
        else:
            # use a heuristic value as the init range
            initrange = 0.001
            self.HN_weight.weight.data.uniform_(-initrange, initrange)
            self.HN_weight.bias.data.zero_()

        if self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
            initrange = np.sqrt(1 / self.base_input_dim)
            self.HN_bias.weight.data.zero_()
            self.HN_bias.bias.data.uniform_(-initrange, initrange)
        else:
            self.HN_bias.weight.data.zero_()
            self.HN_bias.bias.data.zero_()
    
    def forward(self, context_embedding):
        weight = self.HN_weight(context_embedding)
        bias = self.HN_bias(context_embedding)
        return weight, bias
