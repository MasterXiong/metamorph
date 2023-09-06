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


class MLPModel(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(MLPModel, self).__init__()
        self.model_args = cfg.MODEL.MLP
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.limb_obs_size = limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.limb_out_dim = out_dim // self.seq_len

        context_obs_size = obs_space["context"].shape[0] // self.seq_len
        context_encoder_dim = [context_obs_size] + [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE for _ in range(cfg.MODEL.TRANSFORMER.HN_CONTEXT_LAYER_NUM)]
        HN_input_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE

        # set the input and output layer
        if self.model_args.HN_INPUT:
            print ('use HN for input layer')
            # self.context_encoder_for_input = tu.make_mlp_default(context_encoder_dim)
            # self.context_encoder_for_input[0].weight.data.uniform_(-0.01, 0.01)
            # self.context_encoder_for_input[0].bias.data.uniform_(-0.01, 0.01)
            # self.hnet_input_weight = nn.Linear(HN_input_dim, limb_obs_size * self.model_args.HIDDEN_DIM)
            # self.hnet_input_bias = nn.Linear(HN_input_dim, self.model_args.HIDDEN_DIM)
            # self.hnet_input_weight = nn.Embedding(HN_input_dim, limb_obs_size * self.model_args.HIDDEN_DIM)
            # self.hnet_input_bias = nn.Embedding(HN_input_dim, self.model_args.HIDDEN_DIM)

            initrange = 0.04
            self.hnet_input_weight = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.seq_len, self.limb_obs_size, self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hnet_input_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.seq_len, self.model_args.HIDDEN_DIM).uniform_(-initrange / self.seq_len, initrange / self.seq_len))

            # initrange = 0.04
            # self.hnet_input_weight.weight.data.zero_()
            # self.hnet_input_weight.weight.data.uniform_(-0.001, 0.001)
            # # self.hnet_input_weight.bias.data.uniform_(-initrange, initrange)
            # self.hnet_input_weight.bias.data.zero_()
            # # self.hnet_input_bias.weight.data.zero_()
            # self.hnet_input_bias.weight.data.uniform_(-0.001, 0.001)
            # self.hnet_input_bias.bias.data.zero_()
        elif self.model_args.PER_NODE_EMBED:
            print ('independent input weights for each node')
            initrange = 0.04
            self.limb_embed_weights = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            # init_weight = torch.zeros(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange)
            # self.limb_embed_weights = nn.Parameter(torch.stack([init_weight for _ in cfg.ENV.WALKERS]))
            # init_bias = torch.zeros(self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange)
            # self.limb_embed_bias = nn.Parameter(torch.stack([init_bias for _ in cfg.ENV.WALKERS]))
        elif self.model_args.SHARE_INPUT:
            print ('use shared input layer')
            self.shared_input_layer = nn.Linear(self.limb_obs_size, self.model_args.HIDDEN_DIM)
            # initrange = 1. / np.sqrt(obs_space["proprioceptive"].shape[0])
            # self.shared_input_layer.weight.data.uniform_(-initrange, initrange)
            # self.shared_input_layer.bias.data.uniform_(-initrange, initrange)
        else:
            self.input_layer = nn.Linear(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM)
            # if self.model_args.SHARE_INPUT_INIT:
            #     for i in range(1, self.seq_len):
            #         self.input_layer.weight.data[:, (i * self.limb_obs_size):((i + 1) * self.limb_obs_size)].copy_(self.input_layer.weight.data[:, :self.limb_obs_size]) 

        if self.model_args.HN_OUTPUT:
            print ('use HN for output layer')
            # self.context_encoder_for_output = tu.make_mlp_default(context_encoder_dim)
            # self.hnet_output_weight = nn.Linear(HN_input_dim, self.model_args.HIDDEN_DIM * self.limb_out_dim)
            # self.hnet_output_bias = nn.Linear(HN_input_dim, self.limb_out_dim)
            # self.hnet_output_weight = nn.Embedding(HN_input_dim, self.model_args.HIDDEN_DIM * self.limb_out_dim)
            # self.hnet_output_bias = nn.Embedding(HN_input_dim, self.limb_out_dim)

            initrange = 1. / 16
            self.hnet_output_weight = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.seq_len, self.model_args.HIDDEN_DIM, self.limb_out_dim).uniform_(-initrange, initrange))
            self.hnet_output_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.seq_len, self.limb_out_dim).uniform_(-initrange, initrange))

            # initrange = 1. / 16
            # # self.hnet_output_weight.weight.data.zero_()
            # self.hnet_output_weight.weight.data.uniform_(-0.001, 0.001)
            # self.hnet_output_weight.bias.data.zero_()
            # self.hnet_output_bias.weight.data.uniform_(-0.001, 0.001)
            # self.hnet_output_bias.bias.data.zero_()
        elif self.model_args.PER_NODE_DECODER:
            print ('independent output weights for each node')
            initrange = 1. / 16
            self.limb_output_weights = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM, out_dim).uniform_(-initrange, initrange))
            self.limb_output_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), out_dim).uniform_(-initrange, initrange))
            # output_weight = torch.zeros(self.model_args.HIDDEN_DIM, out_dim).uniform_(-initrange, initrange)
            # self.limb_output_weights = nn.Parameter(torch.stack([output_weight for _ in cfg.ENV.WALKERS]))
            # output_bias = torch.zeros(out_dim).uniform_(-initrange, initrange)
            # self.limb_output_bias = nn.Parameter(torch.stack([output_bias for _ in cfg.ENV.WALKERS]))
        else:
            self.output_layer = nn.Linear(self.model_args.HIDDEN_DIM, out_dim)
        
        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
            hidden_dims = [self.model_args.HIDDEN_DIM + self.hfield_encoder.obs_feat_dim] + [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM - 1)]
            self.hidden_layers = tu.make_mlp_default(hidden_dims)
        else:
            hidden_dims = [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM)]
            self.hidden_layers = tu.make_mlp_default(hidden_dims)

        if self.model_args.PER_ROBOT_HIDDEN_LAYERS:
            initrange = 1. / 16
            self.hidden_weights_1 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM, self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hidden_bias_1 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hidden_weights_2 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM, self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hidden_bias_2 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hidden_weights_3 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM, self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.hidden_bias_3 = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))

        if self.model_args.NORM == 'BN':
            # TODO: consider hfield input
            self.batch_norm = nn.BatchNorm1d(self.model_args.HIDDEN_DIM)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, unimal_ids=None):

        batch_size = obs.shape[0]
        # input layer
        if self.model_args.HN_INPUT:

            # context_embedding = self.context_encoder_for_input(obs_context.reshape(batch_size, self.seq_len, -1))
            # input_weight = self.hnet_input_weight(context_embedding).reshape(batch_size, self.seq_len, self.limb_obs_size, self.model_args.HIDDEN_DIM)
            # input_bias = self.hnet_input_bias(context_embedding)
            input_weight = self.hnet_input_weight[unimal_ids]
            input_bias = self.hnet_input_bias[unimal_ids]

            obs = obs.reshape(batch_size, self.seq_len, -1)
            embedding = (obs[:, :, :, None] * input_weight).sum(dim=-2) + input_bias
            # setting zero-padding limbs' values to 0
            # embedding shape: batch_size * limb_num * hidden_layer_dim
            embedding = embedding * (1. - obs_mask.float())[:, :, None]

            # aggregate all limbs' embedding
            embedding = embedding.sum(dim=1)
            embedding = F.relu(embedding)
            
            # context_embedding = self.context_encoder_for_input(obs_context.reshape(batch_size, self.seq_len, -1))
            # obs = obs.reshape(batch_size, self.seq_len, -1)
            # # print (obs[0, obs_mask[0], :6])
            # input_weight = self.hnet_input_weight(context_embedding).reshape(batch_size, self.seq_len, self.limb_obs_size, self.model_args.HIDDEN_DIM)
            # input_bias = self.hnet_input_bias(context_embedding)
            # if self.model_args.SQUASH_HN_OUTPUT:
            #     input_weight = F.tanh(input_weight) * self.model_args.SQUASH_SCALE
            #     input_bias = F.tanh(input_bias) * self.model_args.SQUASH_SCALE
            # # self.input_weight = input_weight.detach().clone()
            # embedding = (obs[:, :, :, None] * input_weight).sum(dim=-2) + input_bias
            # # setting zero-padding limbs' values to 0
            # # embedding shape: batch_size * limb_num * hidden_layer_dim
            # # print (embedding[0, obs_mask[0], :6])
            # embedding = embedding * (1. - obs_mask.float())[:, :, None]
            # # print (embedding[0, obs_mask[0], :6])
            # # TODO: take the mean over truly existing limbs
            # # limb_num = self.seq_len - obs_mask.float().sum(dim=1)
            # # embedding = embedding.sum(dim=1) / limb_num[:, None]
            # if self.model_args.RELU_AFTER_AGG:
            #     if self.model_args.AGG_FUNCTION == 'sum':
            #         embedding = embedding.sum(dim=1)
            #     else:
            #         # scale with the number of existing limbs
            #         embedding = embedding.sum(dim=1) / (self.seq_len - obs_mask.float().sum(dim=1))[:, None]
            #     embedding = F.relu(embedding)
            # else:
            #     if self.model_args.NORM == 'BN':
            #         embedding = self.batch_norm(embedding)
            #     embedding = F.relu(embedding)
            #     if self.model_args.AGG_FUNCTION == 'sum':
            #         embedding = embedding.sum(dim=1)
            #     else:
            #         # scale with the number of existing limbs
            #         embedding = embedding.sum(dim=1) / (self.seq_len - obs_mask.float().sum(dim=1))[:, None]
                    # embedding = embedding.sum(dim=1) * self.seq_len / (self.seq_len - obs_mask.float().sum(dim=1))[:, None]
            # self.hidden_activation = embedding.detach().clone()
        elif self.model_args.PER_NODE_EMBED:
            obs = obs.reshape(batch_size, self.seq_len, -1) * (1. - obs_mask.float())[:, :, None]
            obs = obs.reshape(batch_size, -1)
            embedding = obs[:, :, None] * self.limb_embed_weights[unimal_ids, :, :]
            embedding = embedding.sum(dim=-2, keepdim=False) + self.limb_embed_bias[unimal_ids, :]
            embedding = F.relu(embedding)
        elif self.model_args.SHARE_INPUT:
            obs = obs.reshape(batch_size, self.seq_len, -1)
            embedding = self.shared_input_layer(obs)
            embedding = F.relu(embedding)
            embedding = embedding.mean(dim=1)
        else:
            # zero-padding limbs won't have zero value due to vector normalization. 
            # Need to explicitly set them as 0 to avoid their influence on the hidden layer computation
            obs = obs.reshape(batch_size, self.seq_len, -1) * (1. - obs_mask.float())[:, :, None]
            obs = obs.reshape(batch_size, -1)
            embedding = self.input_layer(obs)
            # rescale by limb number
            # embedding = embedding * cfg.MODEL.MAX_LIMBS / (1. - obs_mask.float()).sum(dim=1)[:, None]
            # self.input_weight = self.input_layer.weight.data.detach().clone()
            # self.hidden_activation = embedding.detach().clone()
            embedding = F.relu(embedding)
            # self.embedding_record = embedding.clone().detach().cpu().numpy()
        
        if self.model_args.SCALE_BY_LIMB_NUM:
            embedding = embedding * cfg.MODEL.MAX_LIMBS / (1. - obs_mask.float()).sum(dim=1)[:, None]

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            hfield_embedding = self.hfield_encoder(obs_env["hfield"])
            embedding = torch.cat([embedding, hfield_embedding], 1)
        
        # hidden layers
        if not self.model_args.PER_ROBOT_HIDDEN_LAYERS:
            embedding = self.hidden_layers(embedding)
        else:
            embedding = (embedding[:, :, None] * self.hidden_weights_1[unimal_ids]).sum(dim=-2) + self.hidden_bias_1[unimal_ids]
            embedding = F.relu(embedding)
            embedding = (embedding[:, :, None] * self.hidden_weights_2[unimal_ids]).sum(dim=-2) + self.hidden_bias_2[unimal_ids]
            embedding = F.relu(embedding)
            embedding = (embedding[:, :, None] * self.hidden_weights_3[unimal_ids]).sum(dim=-2) + self.hidden_bias_3[unimal_ids]
            embedding = F.relu(embedding)

        # output layer
        # if self.model_args.MODE == 'HN':
        if self.model_args.HN_OUTPUT:
            # context_embedding = self.context_encoder_for_output(obs_context.reshape(batch_size, self.seq_len, -1))
            # context_embedding = torch.zeros(batch_size, self.seq_len, len(cfg.ENV.WALKERS) * self.seq_len)
            # for i, idx in enumerate(unimal_ids):
            #     for j in range(self.seq_len):
            #         context_embedding[i, j, idx * self.seq_len + j] = 1.
            # output_weight = self.hnet_output_weight(context_embedding).reshape(batch_size, self.seq_len, self.model_args.HIDDEN_DIM, self.limb_out_dim)
            # output_bias = self.hnet_output_bias(context_embedding)
            output_weight = self.hnet_output_weight[unimal_ids]
            output_bias = self.hnet_output_bias[unimal_ids]
            output = (embedding[:, None, :, None] * output_weight).sum(dim=-2) + output_bias
            # output_check = torch.stack([(embedding[:, :, None] * output_weight[:, i, :, :]).sum(axis=-2) + output_bias[:, i] for i in range(self.seq_len)], dim=1)
            # print ((output != output_check).sum())
            output = output.reshape(batch_size, -1)

        elif self.model_args.PER_NODE_DECODER:
            output = (embedding[:, :, None] * self.limb_output_weights[unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_output_bias[unimal_ids]

        else:
            output = self.output_layer(embedding)

        return output, None, 0.

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
        if self.model_args.PER_NODE_EMBED:
            print ('independent weights for each node')
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed_weights = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), limb_obs_size, self.d_model).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), self.d_model))
        else:
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
        if "visitation" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["visitation"].shape[0])

        if self.ext_feat_fusion == "late":
            decoder_input_dim += self.hfield_encoder.obs_feat_dim
        self.decoder_input_dim = decoder_input_dim

        # self.decoder = nn.Linear(decoder_input_dim, decoder_out_dim)
        if self.model_args.PER_NODE_DECODER:
            # only support a single output layer
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder_weights = torch.zeros(decoder_input_dim, decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_weights = self.decoder_weights.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1, 1)
            self.decoder_weights = nn.Parameter(self.decoder_weights)
            self.decoder_bias = torch.zeros(decoder_out_dim).uniform_(-initrange, initrange)
            self.decoder_bias = self.decoder_bias.repeat(self.seq_len, len(cfg.ENV.WALKERS), 1)
            self.decoder_bias = nn.Parameter(self.decoder_bias)
            # self.decoder_weights = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), decoder_input_dim, decoder_out_dim).uniform_(-initrange, initrange))
            # self.decoder_bias = nn.Parameter(torch.zeros(self.seq_len, len(cfg.ENV.WALKERS), decoder_out_dim))
        else:
            self.decoder = tu.make_mlp_default(
                [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim],
                final_nonlinearity=False,
            )

        if self.model_args.FIX_ATTENTION:
            print ('use fix attention')
            # the network to generate context embedding from the morphology context
            if self.model_args.CONTEXT_AS_FA_INPUT:
                context_obs_size = obs_space["context"].shape[0] // self.seq_len
                self.context_embed_attention = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            else:
                self.context_embed_attention = nn.Linear(limb_obs_size, self.model_args.CONTEXT_EMBED_SIZE)

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
                modules = [nn.ReLU()]
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_attention = nn.Sequential(*modules)

            # if self.model_args.RNN_CONTEXT:
            #     context_encoder_layers = TransformerEncoderLayerResidual(
            #         self.model_args.CONTEXT_EMBED_SIZE,
            #         self.model_args.NHEAD,
            #         self.model_args.DIM_FEEDFORWARD,
            #         self.model_args.DROPOUT,
            #     )
            #     self.rnn_context_encoder_FA = TransformerEncoder(
            #         context_encoder_layers, 1, norm=None,
            #     )

            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                self.context_hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
                self.context_compress = nn.Sequential(
                    nn.Linear(self.model_args.EXT_HIDDEN_DIMS[-1] + self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                    nn.ReLU(), 
                )

        if self.model_args.HYPERNET:
            print ('use HN')
            # the network to generate context embedding from the morphology context
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
            self.context_embed_HN = nn.Linear(context_obs_size, self.model_args.CONTEXT_EMBED_SIZE)
            
            if self.model_args.HN_CONTEXT_ENCODER == 'linear':
                modules = [nn.ReLU()]
                for _ in range(self.model_args.HN_CONTEXT_LAYER_NUM):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_HN = nn.Sequential(*modules)
            elif self.model_args.HN_CONTEXT_ENCODER == 'transformer':
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.context_encoder_HN = TransformerEncoder(
                    context_encoder_layers, self.model_args.HN_CONTEXT_LAYER_NUM, norm=None,
                )

            HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE

            self.hnet_embed_weight = nn.Linear(HN_input_dim, limb_obs_size * self.d_model)
            self.hnet_embed_bias = nn.Linear(HN_input_dim, self.d_model)

            self.decoder_dims = [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim]

            self.hnet_decoder_weight = []
            self.hnet_decoder_bias = []
            for i in range(len(self.decoder_dims) - 1):
                layer_w = nn.Linear(HN_input_dim, self.decoder_dims[i] * self.decoder_dims[i + 1])
                self.hnet_decoder_weight.append(layer_w)
                layer_b = nn.Linear(HN_input_dim, self.decoder_dims[i + 1])
                self.hnet_decoder_bias.append(layer_b)
            self.hnet_decoder_weight = nn.ModuleList(self.hnet_decoder_weight)
            self.hnet_decoder_bias = nn.ModuleList(self.hnet_decoder_bias)

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
            # self.context_encoder_PE = nn.Sequential(
            #     nn.ReLU(), 
            #     nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE)
            # )
            if self.model_args.RNN_CONTEXT:
                # self.rnn_context_encoder = nn.GRU(
                #     input_size=self.model_args.CONTEXT_EMBED_SIZE, 
                #     hidden_size=self.model_args.CONTEXT_EMBED_SIZE, 
                #     num_layers=1, 
                #     bidirectional=False, 
                # )
                # self.context_linear_encoder = nn.Sequential(
                #     nn.ReLU(), 
                #     nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE), 
                # )
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.rnn_context_encoder = TransformerEncoder(
                    context_encoder_layers, 1, norm=None,
                )
                # self.rnn_output_encoder = nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE)
        
            # self.compress_embedding_PE = nn.Linear(2 * self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE)

        if self.model_args.USE_SWAT_PE:
            self.swat_PE_encoder = SWATPEEncoder(self.d_model, self.seq_len)
        if self.model_args.USE_SEPARATE_PE:
            self.separate_PE_encoder = SeparatePEEncoder(self.d_model, self.seq_len)
        if self.model_args.USE_SEMANTIC_PE:
            self.semantic_PE_encoder = SemanticEncoding(self.d_model, self.seq_len)

        self.dropout = nn.Dropout(p=0.1)

        self.init_weights()
        self.count = 0

    def init_weights(self):
        # init obs embedding
        if not self.model_args.PER_NODE_EMBED:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.limb_embed.weight.data.uniform_(-initrange, initrange)
        # init decoder
        if not self.model_args.PER_NODE_DECODER:
            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            self.decoder[-1].bias.data.zero_()
            self.decoder[-1].weight.data.uniform_(-initrange, initrange)

        if self.model_args.FIX_ATTENTION:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_attention.weight.data.uniform_(-initrange, initrange)

            # force the attention weight to be uniform at the beginning
            # self.context_encoder_attention[-2].weight.data.zero_()
            # self.context_encoder_attention[-2].bias.data.zero_()

        if self.model_args.HYPERNET:
            initrange = cfg.MODEL.TRANSFORMER.HN_EMBED_INIT
            self.context_embed_HN.weight.data.uniform_(-initrange, initrange)

            # initialize the hypernet following Jake's paper
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()

            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            for i in range(len(self.hnet_decoder_weight)):
                self.hnet_decoder_weight[i].weight.data.zero_()
                self.hnet_decoder_weight[i].bias.data.uniform_(-initrange, initrange)
                self.hnet_decoder_bias[i].weight.data.zero_()
                self.hnet_decoder_bias[i].bias.data.zero_()

        if self.model_args.CONTEXT_PE:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_PE.weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, unimal_ids=None):
        # (num_limbs, batch_size, limb_obs_size) -> (num_limbs, batch_size, d_model)
        _, batch_size, limb_obs_size = obs.shape

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["hfield"])
        if "visitation" in cfg.ENV.KEYS_TO_KEEP:
            # (batch_size, embed_size)
            hfield_obs = self.hfield_encoder(obs_env["visitation"])

        if self.ext_feat_fusion in ["late"]:
            hfield_obs = hfield_obs.repeat(self.seq_len, 1)
            hfield_obs = hfield_obs.reshape(self.seq_len, batch_size, -1)

        if self.model_args.FIX_ATTENTION:
            if self.model_args.CONTEXT_AS_FA_INPUT:
                context_embedding_attention = self.context_embed_attention(obs_context)
            else:
                context_embedding_attention = self.context_embed_attention(obs)

            # if self.model_args.RNN_CONTEXT:
            #     mask = torch.cat([morphology_info['node_path_mask'] for _ in range(self.model_args.NHEAD)], 0)
            #     context_embedding_attention = self.rnn_context_encoder_FA(
            #         context_embedding_attention, 
            #         mask=mask, 
            #     )

            if self.model_args.CONTEXT_ENCODER == 'transformer':
                context_embedding_attention = self.context_encoder_attention(
                    context_embedding_attention, 
                    src_key_padding_mask=obs_mask, 
                    morphology_info=morphology_info)
            else:
                context_embedding_attention = self.context_encoder_attention(context_embedding_attention)

            if self.model_args.HFIELD_IN_FIX_ATTENTION:
                hfield_embedding = self.context_hfield_encoder(obs_env["hfield"])
                hfield_embedding = hfield_embedding.repeat(self.seq_len, 1).reshape(self.seq_len, batch_size, -1)
                context_embedding_attention = torch.cat([context_embedding_attention, hfield_embedding], dim=-1)
                context_embedding_attention = self.context_compress(context_embedding_attention)
                # hfield_embedding = self.hfield_embed(hfield_obs)
                # context_embedding_attention = context_embedding_attention + hfield_embedding

        if self.model_args.HYPERNET:
            context_embedding_HN = self.context_embed_HN(obs_context)
            context_embedding_HN = self.context_encoder_HN(context_embedding_HN)

        if self.model_args.CONTEXT_PE:
            context_embedding_PE = self.context_embed_PE(obs_context)
            # context_embedding_PE = self.context_linear_encoder(context_embedding_PE)
            context_embedding_PE *= math.sqrt(self.model_args.CONTEXT_EMBED_SIZE)
            if self.model_args.RNN_CONTEXT:
                mask = torch.cat([morphology_info['node_path_mask'] for _ in range(self.model_args.NHEAD)], 0)
                context_embedding_PE = self.rnn_context_encoder(
                    context_embedding_PE, 
                    mask=mask, 
                )                
                # # batch_size * seq_len * max_tree_depth * embedding_size
                # context_embedding_PE = self.context_embed_PE(obs_context_path)
                # # add PE
                # context_embedding_PE = self.tree_path_PE(context_embedding_PE)
                # # combine the first two dimensions for RNN processing
                # context_embedding_PE = context_embedding_PE.reshape(-1, self.model_args.MAX_NODE_DEPTH, self.model_args.CONTEXT_EMBED_SIZE)
                # # go through the RNN/TF encoder
                # # context_embedding_PE, _ = self.rnn_context_encoder(context_embedding_PE)
                # mask = morphology_info['node_path_mask'].reshape(-1, self.model_args.MAX_NODE_DEPTH)
                # context_embedding_PE = self.rnn_context_encoder(
                #     context_embedding_PE, 
                #     src_key_padding_mask=mask, 
                # )
                # # select the node's embedding
                # row_idx = torch.from_numpy(np.arange(context_embedding_PE.shape[0])).long()
                # col_idx = morphology_info['node_path_length'].reshape(-1).long()
                # context_embedding_PE = context_embedding_PE[row_idx, col_idx]
                # # linear encoder layer on RNN output
                # context_embedding_PE = self.rnn_output_encoder(context_embedding_PE)
                # # reshape and permute
                # context_embedding_PE = context_embedding_PE.reshape(batch_size, self.seq_len, -1)
                # context_embedding_PE = context_embedding_PE.permute(1, 0, 2)

            context_embedding_PE = self.context_encoder_PE(context_embedding_PE, src_key_padding_mask=obs_mask)
            # context_embedding_PE = self.context_encoder_PE(context_embedding_PE)

        if self.model_args.HYPERNET and self.model_args.HN_EMBED:
            embed_weight = self.hnet_embed_weight(context_embedding_HN).reshape(self.seq_len, batch_size, limb_obs_size, self.d_model)
            embed_bias = self.hnet_embed_bias(context_embedding_HN)
            obs_embed = (obs[:, :, :, None] * embed_weight).sum(dim=-2, keepdim=False) + embed_bias
        else:
            if self.model_args.PER_NODE_EMBED:
                obs_embed = (obs[:, :, :, None] * self.limb_embed_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_embed_bias[:, unimal_ids, :]
            else:
                obs_embed = self.limb_embed(obs)
        
        if self.model_args.EMBEDDING_SCALE:
            obs_embed *= math.sqrt(self.d_model)
        # self.embedding_record = obs_embed.clone()

        attention_maps = None

        # plt.figure()
        # plt.hist(obs_embed.detach().cpu().numpy().ravel(), bins=100, range=(-5., 5.), histtype='step', label='embedding')
        # plt.hist(context_embedding_PE.detach().cpu().numpy().ravel(), bins=100, range=(-5., 5.), histtype='step', label='PE')
        # print (f'obs: mean: {obs_embed.detach().cpu().numpy().ravel().mean():.2f}, std:{obs_embed.detach().cpu().numpy().ravel().std():.2f}')
        # print (f'PE: mean: {context_embedding_PE.detach().cpu().numpy().ravel().mean():.2f}, std:{context_embedding_PE.detach().cpu().numpy().ravel().std():.2f}')
        # plt.savefig(f'figures/rnn_PE_check/{self.count}.png')
        # plt.close()
        # self.count += 1

        # add PE
        if self.model_args.POS_EMBEDDING in ["learnt", "abs"]:
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.CONTEXT_PE:
            obs_embed = obs_embed + context_embedding_PE
            # obs_embed = self.compress_embedding_PE(torch.cat([obs_embed, context_embedding_PE], -1))
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info['traversals'])
        if self.model_args.USE_SEPARATE_PE:
            obs_embed = self.separate_PE_encoder(obs_embed, unimal_ids)
        if self.model_args.USE_SEMANTIC_PE:
            obs_embed = self.semantic_PE_encoder(obs_embed, morphology_info['position_id'])

        # print (f'embedding after PE: mean: {obs_embed.detach().cpu().numpy().ravel().mean():.2f}, std:{obs_embed.detach().cpu().numpy().ravel().std():.2f}')
        # plt.hist(obs_embed.detach().cpu().numpy().ravel(), bins=100, range=(-5., 5.), histtype='step', label='after PE')
        # plt.legend()
        # plt.savefig(f'figures/graph_PE_check/{self.count}.png')
        # plt.close()
        # self.count += 1

        # dropout
        # if self.model_args.EMBEDDING_DROPOUT:
        #     if self.model_args.CONSISTENT_DROPOUT:
        #         # print ('consistent dropout')
        #         if dropout_mask is None:
        #             obs_embed_after_dropout = self.dropout(obs_embed)
        #             # print (obs_embed_after_dropout / obs_embed)
        #             dropout_mask = torch.where(obs_embed_after_dropout == 0., 0., 1.).permute(1, 0, 2)
        #             # print (obs_embed_after_dropout == (obs_embed * dropout_mask.permute(1, 0, 2) / 0.9))
        #             obs_embed = obs_embed_after_dropout
        #         else:
        #             obs_embed = obs_embed * dropout_mask.permute(1, 0, 2) / 0.9
        #     else:
        #         # print ('vanilla dropout')
        #         obs_embed = self.dropout(obs_embed)
        #         # ratio = obs_embed_after_dropout / obs_embed
        #         # print (f'0.9 in ratio: {(ratio == 1. / 0.9).sum()}, 0. in ratio: {(ratio == 0.).sum()}')
        #         # dropout_mask = torch.where(obs_embed == 0., 0., 1.).permute(1, 0, 2)
        #         dropout_mask = 0.
        # else:
        #     dropout_mask = 0.

        if self.model_args.FIX_ATTENTION:
            context_to_base = context_embedding_attention
        else:
            context_to_base = None
        
        if self.model_args.USE_CONNECTIVITY_IN_ATTENTION:
            attn_mask = torch.cat([morphology_info['connectivity'][:, :, :, i] for i in range(4)], 0)
        elif self.model_args.USE_SWAT_RE:
            attn_mask = morphology_info['SWAT_RE']
        else:
            attn_mask = None
        src_key_padding_mask = obs_mask

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                obs_embed, 
                mask=attn_mask, 
                src_key_padding_mask=src_key_padding_mask, 
                context=context_to_base, 
                morphology_info=morphology_info
            )
        
        decoder_input = obs_embed_t
        if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)
        if "visitation" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
            decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        if self.model_args.HYPERNET and self.model_args.HN_DECODER:
            output = decoder_input
            layer_num = len(self.hnet_decoder_weight)
            for i in range(layer_num):
                layer_w = self.hnet_decoder_weight[i](context_embedding_HN).reshape(self.seq_len, batch_size, self.decoder_dims[i], self.decoder_dims[i + 1])
                layer_b = self.hnet_decoder_bias[i](context_embedding_HN)
                output = (output[:, :, :, None] * layer_w).sum(dim=-2, keepdim=False) + layer_b
                if i != (layer_num - 1):
                    output = F.relu(output)
        else:
            if self.model_args.PER_NODE_DECODER:
                output = (decoder_input[:, :, :, None] * self.decoder_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.decoder_bias[:, unimal_ids, :]
            else:
                output = self.decoder(decoder_input)

        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if batch_first:
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            # self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))
            self.pe = nn.Parameter(torch.zeros(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


class SemanticEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(seq_len, 1, d_model))

    def forward(self, x, index):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[index.reshape(-1)].reshape(index.shape[0], index.shape[1], -1).permute(1, 0, 2)
        x = x + pe
        return x


class SWATPEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe_dim = [d_model // len(cfg.MODEL.TRANSFORMER.TRAVERSALS) for _ in cfg.MODEL.TRANSFORMER.TRAVERSALS]
        self.pe_dim[-1] = d_model - self.pe_dim[0] * (len(cfg.MODEL.TRANSFORMER.TRAVERSALS) - 1)
        print (self.pe_dim)
        self.swat_pe = nn.ModuleList([nn.Embedding(seq_len, dim) for dim in self.pe_dim])
        # self.compress_layer = nn.Linear(len(cfg.MODEL.TRANSFORMER.TRAVERSALS) * d_model, d_model)

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
        # x = x + self.compress_layer(embeddings)
        x = x + embeddings
        return x


class SeparatePEEncoder(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # self.pe = nn.Parameter(torch.randn(seq_len, len(cfg.ENV.WALKERS), d_model))
        self.pe = nn.Parameter(torch.zeros(seq_len, len(cfg.ENV.WALKERS), d_model))

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
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


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
        if cfg.MODEL.TYPE == 'transformer':
            self.v_net = TransformerModel(obs_space, 1)
        else:
            if cfg.MODEL.MLP.SINGLE_VALUE:
                self.v_net = MLPValueNetwork(obs_space)
            else:
                self.v_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS)

        if cfg.ENV_NAME == "Unimal-v0":
            if cfg.MODEL.TYPE == 'transformer':
                self.mu_net = TransformerModel(obs_space, 2)
            else:
                self.mu_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS * 2)
            self.num_actions = cfg.MODEL.MAX_LIMBS * 2
        elif cfg.ENV_NAME == 'Modular-v0':
            if cfg.MODEL.TYPE == 'transformer':
                self.mu_net = TransformerModel(obs_space, 1)
            else:
                self.mu_net = MLPModel(obs_space, cfg.MODEL.MAX_LIMBS)
            self.num_actions = cfg.MODEL.MAX_LIMBS
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
        joint_context_index = np.concatenate([np.arange(2, 2 + 9), np.arange(11 + 2, 11 + 2 + 9)]) + 30
        self.context_index = np.concatenate([limb_context_index, joint_context_index])
        print ('context index', self.context_index)

        # hard code fix-range normalization for state inputs
        if cfg.ENV_NAME == "Unimal-v0":
            self.state_norm_index = np.concatenate([np.arange(9), np.array([30, 31, 41, 42])])
            self.state_min = np.array([-1.6, -25, -1, -15, -15, -15, -50, -50, -50, -0.3, -35, -0.3, -35])
            self.state_min = torch.Tensor(self.state_min.reshape(1, 1, -1)).cuda()
            self.state_max = np.array([1.6, 25, 4, 15, 15, 15, 50, 50, 50, 1.3, 35, 1.3, 35])
            self.state_max = torch.Tensor(self.state_max.reshape(1, 1, -1)).cuda()
        elif cfg.ENV_NAME == 'Modular-v0':
            self.state_norm_index = np.arange(12)
            if 'humanoid' in cfg.ENV.WALKER_DIR:
                self.state_min = np.array([-1, -0.4, 0, -10, -10, -10, -1, -40, -1, -np.pi, -np.pi, -np.pi])
                self.state_max = np.array([55, 0.4, 1.5, 10, 10, 10, 1, 40, 1, np.pi, np.pi, np.pi])
            elif 'walker' in cfg.ENV.WALKER_DIR:
                self.state_min = np.array([-1, -1, 0, -10, -10, -10, -1, -60, -1, -np.pi, -np.pi, -np.pi])
                self.state_max = np.array([50, 1, 1.6, 10, 10, 10, 1, 60, 1, np.pi, np.pi, np.pi])
            self.state_min = torch.Tensor(self.state_min.reshape(1, 1, -1)).cuda()
            self.state_max = torch.Tensor(self.state_max.reshape(1, 1, -1)).cuda()

    def forward(self, obs, act=None, return_attention=False, unimal_ids=None, compute_val=True):
        
        # all_start = time.time()
        
        if act is not None:
            batch_size = cfg.PPO.BATCH_SIZE
            batch_size = act.shape[0]
        else:
            # batch_size = cfg.PPO.NUM_ENVS
            batch_size = obs['proprioceptive'].shape[0]

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
        if cfg.MODEL.TRANSFORMER.USE_CONNECTIVITY_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity'].bool()
        if cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity']
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['traversals'] = obs_dict['traversals'].permute(1, 0, 2).long()
        if cfg.MODEL.TRANSFORMER.USE_SWAT_RE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['SWAT_RE'] = obs_dict['SWAT_RE']
        if cfg.MODEL.TRANSFORMER.RNN_CONTEXT:
            morphology_info['node_path_length'] = obs_dict['node_path_length']
            morphology_info['node_path_mask'] = obs_dict['node_path_mask'].bool()
        # semantic for modular robots
        if cfg.MODEL.TRANSFORMER.USE_SEMANTIC_PE:
            morphology_info['position_id'] = obs_dict['position_id'].long()
        
        if len(morphology_info.keys()) == 0:
            morphology_info = None
        # end = time.time()
        # print ('time on connectivity', end - start)

        obs_mask = obs_mask.bool()
        act_mask = act_mask.bool()

        # reshape the obs for transformer input
        if cfg.MODEL.TYPE == 'transformer':
            obs = obs.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
            obs_context = obs_context.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
            if cfg.MODEL.BASE_CONTEXT_NORM == 'fixed':
                obs[:, :, self.context_index] = obs_context.clone()
            if cfg.MODEL.OBS_FIX_NORM:
                normed_obs = obs.clone()
                normed_obs[:, :, self.state_norm_index] = -1. + 2. * (obs[:, :, self.state_norm_index] - self.state_min) / (self.state_max - self.state_min)
                obs = normed_obs

        if compute_val:
            # Per limb critic values
            limb_vals, v_attention_maps = self.v_net(
                obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
                return_attention=return_attention,  
                unimal_ids=unimal_ids, 
            )
            if cfg.MODEL.MLP.SINGLE_VALUE:
                val = limb_vals
            else:
                # Zero out mask values
                limb_vals = limb_vals * (1 - obs_mask.int())
                # Use avg/max to keep the magnitidue same instead of sum
                num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
                val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)
        else:
            val, v_attention_maps = 0., None

        mu, mu_attention_maps = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, 
            unimal_ids=unimal_ids, 
        )
        if cfg.PPO.TANH == 'mean':
            mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        # all_end = time.time()
        # print ('full forward time', all_end - all_start)

        if act is not None:
            logp = pi.log_prob(act)
            logp[act_mask] = 0.0
            self.limb_logp = logp
            logp = logp.sum(-1, keepdim=True)
            entropy = pi.entropy()
            entropy[act_mask] = 0.0
            entropy = entropy.mean()
            return val, pi, logp, entropy
        else:
            if return_attention:
                return val, pi, v_attention_maps, mu_attention_maps
            else:
                return val, pi, None, None


class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs, return_attention=False, unimal_ids=None, compute_val=True):
        val, pi, v_attention_maps, mu_attention_maps = self.ac(obs, return_attention=return_attention, unimal_ids=unimal_ids, compute_val=compute_val)
        self.pi = pi
        if not cfg.DETERMINISTIC:
            act = pi.sample()
        else:
            act = pi.loc
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        self.limb_logp = logp
        logp = logp.sum(-1, keepdim=True)
        self.v_attention_maps = v_attention_maps
        self.mu_attention_maps = mu_attention_maps
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs, unimal_ids=None):
        val, _, _, _ = self.ac(obs, unimal_ids=unimal_ids)
        return val
