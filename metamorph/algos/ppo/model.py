import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual

import time
import matplotlib.pyplot as plt


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, obs_space, decoder_out_dim):
        super(TransformerModel, self).__init__()

        self.decoder_out_dim = decoder_out_dim

        self.model_args = cfg.MODEL.TRANSFORMER
        self.seq_len = cfg.MODEL.MAX_LIMBS
        # Embedding layer for per limb obs
        limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE
        self.limb_embed = nn.Linear(limb_obs_size, self.d_model)
        self.ext_feat_fusion = self.model_args.EXT_MIX

        if self.model_args.POS_EMBEDDING == "learnt":
            print ('use PE learnt')
            seq_len = self.seq_len
            self.pos_embedding = PositionalEncoding(self.d_model, seq_len)
        elif self.model_args.POS_EMBEDDING == "abs":
            print ('use PE abs')
            self.pos_embedding = PositionalEncoding1D(self.d_model, self.seq_len)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayerResidual(
            cfg.MODEL.LIMB_EMBED_SIZE,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.model_args.NLAYERS, norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        # Task based observation encoder
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        self.decoder = tu.make_mlp_default(
            [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
            final_nonlinearity=False,
        )

        if self.model_args.FIX_ATTENTION:
            print ('use fix attention')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_attention = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            if self.model_args.CONTEXT_ENCODER == 'transformer':
                print ('use transformer context encoder')
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_attention = TransformerEncoder(
                    context_encoder_layers, self.model_args.CONTEXT_LAYER, norm=None,
                )
            else: # MLP context encoder
                print ('use MLP context encoder')
                modules = []
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules.append(nn.ReLU())
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                self.context_encoder_attention = nn.Sequential(*modules)
                # self.context_encoder_attention = nn.Sequential(
                #     nn.ReLU(), # add a nonlinear layer after linear embedding first
                #     nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                #     nn.ReLU(), 
                #     nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                # )

        if self.model_args.HYPERNET:
            print ('use HN')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_HN = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            context_encoder_layers = TransformerEncoderLayerResidual(
                self.model_args.CONTEXT_EMBED_SIZE,
                self.model_args.NHEAD,
                self.model_args.DIM_FEEDFORWARD,
                self.model_args.DROPOUT,
            )
            self.context_encoder_HN = TransformerEncoder(
                context_encoder_layers, 1, norm=None,
            )

            self.hnet_embed_weight = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, limb_obs_size * self.d_model)
            self.hnet_embed_bias = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.d_model)

            self.hnet_weight = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, decoder_input_dim * decoder_out_dim)
            self.hnet_bias = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, decoder_out_dim)

        if self.model_args.CONTEXT_PE:
            print ('use context PE')
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_PE = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            context_encoder_layers = TransformerEncoderLayerResidual(
                self.model_args.CONTEXT_EMBED_SIZE,
                self.model_args.NHEAD,
                self.model_args.DIM_FEEDFORWARD,
                self.model_args.DROPOUT,
            )
            self.context_encoder_PE = TransformerEncoder(
                context_encoder_layers, 1, norm=None,
            )
        
        if self.model_args.USE_SWAT_PE:
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)
        if self.model_args.USE_SEPARATE_PE:
            self.separate_PE_encoder = SeparatePEEncoder(self.d_model, self.seq_len)

        self.dropout = nn.Dropout(p=0.1)

        if self.model_args.CONTEXT_DROPOUT:
            self.context_dropout = nn.Dropout(p=0.1)

        if self.model_args.USE_NODE_DEPTH:
            self.node_depth_embed = nn.Linear(self.model_args.MAX_NODE_DEPTH, self.model_args.CONTEXT_EMBED_SIZE)

        self.init_weights()

    def init_weights(self):
        # init obs embedding
        initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
        self.decoder[-1].bias.data.zero_()
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

        if self.model_args.FIX_ATTENTION:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_attention.weight.data.uniform_(-initrange, initrange)

        if self.model_args.HYPERNET:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_HN.weight.data.uniform_(-initrange, initrange)

            # initialize the hypernet following Jake's paper
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()

            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.hnet_weight.weight.data.zero_()
            self.hnet_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_bias.weight.data.zero_()
            self.hnet_bias.bias.data.zero_()

        if self.model_args.CONTEXT_PE:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_PE.weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        _, batch_size, limb_obs_size = obs.shape

        if self.model_args.FIX_ATTENTION:
            context_embedding_attention = self.context_embed_attention(obs_context)
            if self.model_args.USE_NODE_DEPTH:
                context_embedding_attention = context_embedding_attention + self.node_depth_embed(morphology_info['node_depth'])
            if self.model_args.CONTEXT_DROPOUT:
                context_embedding_attention = self.context_dropout(context_embedding_attention)
            if self.model_args.CONTEXT_ENCODER == 'transformer':
                context_embedding_attention = self.context_encoder_attention(
                    context_embedding_attention, 
                    src_key_padding_mask=obs_mask, 
                    morphology_info=morphology_info)
            else:
                context_embedding_attention = self.context_encoder_attention(context_embedding_attention)

        if self.model_args.HYPERNET:
            context_embedding_HN = self.context_embed_HN(obs_context)
            # TODO: use morphology_info in HN or not
            context_embedding_HN = self.context_encoder_HN(context_embedding_HN, src_key_padding_mask=obs_mask)

        if self.model_args.CONTEXT_PE:
            context_embedding_PE = self.context_embed_PE(obs_context)
            context_embedding_PE = self.context_encoder_PE(context_embedding_PE, src_key_padding_mask=obs_mask)

        if not self.model_args.HYPERNET:
            obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        else:
            embed_weight = self.hnet_embed_weight(context_embedding_HN).reshape(self.seq_len, batch_size, limb_obs_size, self.d_model)
            embed_bias = self.hnet_embed_bias(context_embedding_HN)
            obs_embed = (obs[:, :, :, None] * embed_weight).sum(dim=-2, keepdim=False) + embed_bias
            # obs_embed = obs_embed * math.sqrt(self.d_model)

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        attention_maps = None

        # add PE
        if self.model_args.POS_EMBEDDING in ["learnt", "abs"] and self.model_args.PE_POSITION == 'base':
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.CONTEXT_PE:
            obs_embed = obs_embed + context_embedding_PE
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info['traversals'])
        if self.model_args.USE_SEPARATE_PE:
            obs_embed = self.separate_PE_encoder(obs_embed, unimal_ids)

        # code for dropout test
        if self.model_args.EMBEDDING_DROPOUT:
            if self.model_args.CONSISTENT_DROPOUT:
                # print ('consistent dropout')
                if dropout_mask is None:
                    obs_embed_after_dropout = self.dropout(obs_embed)
                    # print (obs_embed_after_dropout / obs_embed)
                    dropout_mask = torch.where(obs_embed_after_dropout == 0., 0., 1.).permute(1, 0, 2)
                    # print (obs_embed_after_dropout == (obs_embed * dropout_mask.permute(1, 0, 2) / 0.9))
                    obs_embed = obs_embed_after_dropout
                else:
                    obs_embed = obs_embed * dropout_mask.permute(1, 0, 2) / 0.9
            else:
                # print ('vanilla dropout')
                obs_embed = self.dropout(obs_embed)
                # ratio = obs_embed_after_dropout / obs_embed
                # print (f'0.9 in ratio: {(ratio == 1. / 0.9).sum()}, 0. in ratio: {(ratio == 0.).sum()}')
                # dropout_mask = torch.where(obs_embed == 0., 0., 1.).permute(1, 0, 2)
                dropout_mask = 0.
        else:
            dropout_mask = 0.

        if self.model_args.FIX_ATTENTION:
            context_to_base = context_embedding_attention
        else:
            context_to_base = None

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, 
                src_key_padding_mask=obs_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, 
                src_key_padding_mask=obs_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )

        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        if not self.model_args.HYPERNET:
            output = self.decoder(decoder_input)
        else:
            decoder_weight = self.hnet_weight(context_embedding_HN).reshape(self.seq_len, batch_size, self.d_model, self.decoder_out_dim)
            decoder_bias = self.hnet_bias(context_embedding_HN)
            output = (decoder_input[:, :, :, None] * decoder_weight).sum(dim=-2, keepdim=False) + decoder_bias
        
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps, dropout_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if cfg.MODEL.TRANSFORMER.PE_POSITION == 'base':
            self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))
        else:
            self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model) / math.sqrt(cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.swat_pe = nn.ModuleList([nn.Embedding(seq_len, d_model) for _ in cfg.MODEL.TRANSFORMER.TRAVERSALS])
        self.compress_layer = nn.Linear(len(cfg.MODEL.TRANSFORMER.TRAVERSALS) * d_model, d_model)

    def forward(self, x, indexes):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        embeddings = []
        batch_size = x.size(1)
        for i in range(len(cfg.MODEL.TRANSFORMER.TRAVERSALS)):
            idx = indexes[:, :, i]
            pe = self.swat_pe[i](idx)
            embeddings.append(pe)
        embeddings = torch.cat(embeddings, dim=-1)
        x = x + self.compress_layer(embeddings)
        return x


class SeparatePEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(seq_len, len(cfg.ENV.WALKERS), d_model))

    def forward(self, x, unimal_ids):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, unimal_ids, :]
        return x


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS
        self.encoder = tu.make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.v_net = TransformerModel(obs_space, 1)

        if cfg.ENV_NAME == "Unimal-v0":
            self.mu_net = TransformerModel(obs_space, 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2
        else:
            raise ValueError("Unsupported ENV_NAME")

        if cfg.MODEL.ACTION_STD_FIXED:
            log_std = np.log(cfg.MODEL.ACTION_STD)
            self.log_std = nn.Parameter(
                log_std * torch.ones(1, self.num_actions), requires_grad=False,
            )
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))
        
        # hard code the index of context features if they are included in proprioceptive features
        limb_context_index = np.arange(13, 13 + 17)
        # two joints features for each node
        joint_context_index = np.concatenate([np.arange(2, 2 + 9), np.arange(11 + 2, 11 + 2 + 9)])
        self.context_index = np.concatenate([limb_context_index, joint_context_index])

    def forward(self, obs, act=None, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        
        # all_start = time.time()
        
        if act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
        else:
            batch_size = cfg.PPO.NUM_ENVS

        obs_env = {k: obs[k] for k in cfg.ENV.KEYS_TO_KEEP}
        if "obs_padding_cm_mask" in obs:
            obs_cm_mask = obs["obs_padding_cm_mask"]
        else:
            obs_cm_mask = None
        obs_dict = obs
        obs, obs_mask, act_mask, obs_context, edges = (
            obs["proprioceptive"],
            obs["obs_padding_mask"],
            obs["act_padding_mask"],
            obs["context"], 
            obs["edges"], 
        )

        # start = time.time()
        morphology_info = {}
        if cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            # # generate connectivity features of 3d for each node pair: batch_size * seq_len * seq_len * feat_dim
            # # 0: identity indicator; 1: parent; 2: child
            morphology_info['connectivity'] = obs_dict['connectivity']
        if cfg.MODEL.TRANSFORMER.USE_NODE_DEPTH:
            # (batch_size, seq_len, max_node_depth) ->(seq_len, batch_size, max_node_depth)
            morphology_info['node_depth'] = obs_dict['node_depth'].permute(1, 0, 2)
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['traversals'] = obs_dict['traversals'].permute(1, 0, 2).long()
        
        if len(morphology_info.keys()) == 0:
            morphology_info = None
        # end = time.time()
        # print ('time on connectivity', end - start)

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        obs_context = obs_context.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        if cfg.MODEL.BASE_CONTEXT_NORM == 'fixed':
            obs[:, :, self.context_index] = obs_context.clone()
        # print (obs[:, :, self.context_index].min(), obs[:, :, self.context_index].max())
        # Per limb critic values
        limb_vals, v_attention_maps, dropout_mask_v = self.v_net(
            obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, dropout_mask=dropout_mask_v, 
            unimal_ids=unimal_ids
        )
        # Zero out mask values
        limb_vals = limb_vals * (1 - obs_mask.int())
        # Use avg/max to keep the magnitidue same instead of sum
        num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
        val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)

        mu, mu_attention_maps, dropout_mask_mu = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, dropout_mask=dropout_mask_mu, 
            unimal_ids=unimal_ids
        )
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        # all_end = time.time()
        # print ('full forward time', all_end - all_start)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy, dropout_mask_v, dropout_mask_mu
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v, dropout_mask_mu
            else:
                return val, pi, None, None, dropout_mask_v, dropout_mask_mu


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        val, pi, _, _, dropout_mask_v, dropout_mask_mu = self.ac(obs, return_attention=return_attention, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids)
        if not cfg.DETERMINISTIC:
            act = pi.sample()
        else:
            act = pi.loc
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        return val, act, logp, dropout_mask_v, dropout_mask_mu

    @torch.no_grad()
    def get_value(self, obs, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        val, _, _, _, _, _ = self.ac(obs, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids)
        return val
