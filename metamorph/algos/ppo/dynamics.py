import torch
import torch.nn as nn
import torch.optim as optim

import math

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual


class DynamicsModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(DynamicsModel, self).__init__()

        self.seq_len = 12
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.model_args = cfg.MODEL.TRANSFORMER

        self.limb_embed = nn.Linear(input_dim, self.d_model)

        encoder_layers = TransformerEncoderLayerResidual(
            self.d_model,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        self.decoder = tu.make_mlp_default(
            [self.d_model] + self.model_args.DECODER_DIMS + [output_dim],
            final_nonlinearity=False,
        )

        self.init_weights()

    def init_weights(self):
        # init obs embedding
        initrange = self.model_args.EMBED_INIT
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        initrange = self.model_args.DECODER_INIT
        self.decoder[-1].bias.data.zero_()
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs_action_tuple, obs_mask, return_attention=False):

        # obs_action_tuple size: batch_size * seq_len * (obs_size + action_size)
        obs_action_tuple = obs_action_tuple.permute(1, 0, 2)
        
        embedding = self.limb_embed(obs_action_tuple)
        embedding *= math.sqrt(self.d_model)

        embedding = self.transformer_encoder(
            embedding, 
            src_key_padding_mask=obs_mask, 
        )

        output = self.decoder(embedding)
        output = output.permute(1, 0, 2)

        return output


class DynamicsTrainer:

    def __init__(self, input_dim, output_dim):

        self.model = DynamicsModel(input_dim, output_dim)
        self.model.cuda()

        self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=cfg.DYNAMICS.BASE_LR, 
                eps=cfg.DYNAMICS.EPS, 
                weight_decay=cfg.DYNAMICS.WEIGHT_DECAY
            )

    def train_on_batch(self, obs_action_tuple, next_obs, obs_mask):

        next_obs_pred = self.model(obs_action_tuple, obs_mask)
        
        loss = (((next_obs_pred - next_obs) ** 2) * ~obs_mask[:, :, None]).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
