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
                modules = [nn.ReLU()]
                for _ in range(self.model_args.LINEAR_CONTEXT_LAYER):
                    modules.append(nn.Linear(self.model_args.CONTEXT_EMBED_SIZE, self.model_args.CONTEXT_EMBED_SIZE))
                    modules.append(nn.ReLU())
                self.context_encoder_attention = nn.Sequential(*modules)

            if self.model_args.RNN_CONTEXT:
                context_encoder_layers = TransformerEncoderLayerResidual(
                    self.model_args.CONTEXT_EMBED_SIZE,
                    self.model_args.NHEAD,
                    self.model_args.DIM_FEEDFORWARD,
                    self.model_args.DROPOUT,
                )
                self.rnn_context_encoder_FA = TransformerEncoder(
                    context_encoder_layers, 1, norm=None,
                )

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

            if not self.model_args.DEPTH_INPUT_HN:
                HN_input_dim = self.model_args.CONTEXT_EMBED_SIZE
            else:
                HN_input_dim = self.model_args.MAX_NODE_DEPTH

            self.hnet_embed_weight = nn.Linear(HN_input_dim, limb_obs_size * self.d_model)
            self.hnet_embed_bias = nn.Linear(HN_input_dim, self.d_model)

            self.decoder_dims = [decoder_input_dim] + self.model_args.DECODER_DIMS + [decoder_out_dim]

            if self.model_args.HN_DECODER_LAST_LAYER:
                self.shared_decoder_layers = tu.make_mlp_default(self.decoder_dims[:-1])
                self.hnet_decoder_weight = nn.Linear(HN_input_dim, self.decoder_dims[-2] * self.decoder_dims[-1])
                self.hnet_decoder_bias = nn.Linear(HN_input_dim, self.decoder_dims[-1])
            else:
                self.hnet_decoder_weight = []
                self.hnet_decoder_bias = []
                for i in range(len(self.decoder_dims) - 1):
                    layer_w = nn.Linear(HN_input_dim, self.decoder_dims[i] * self.decoder_dims[i + 1])
                    self.hnet_decoder_weight.append(layer_w)
                    layer_b = nn.Linear(HN_input_dim, self.decoder_dims[i + 1])
                    self.hnet_decoder_bias.append(layer_b)
                self.hnet_decoder_weight = nn.ModuleList(self.hnet_decoder_weight)
                self.hnet_decoder_bias = nn.ModuleList(self.hnet_decoder_bias)

            # self.hnet_weight = nn.Linear(HN_input_dim, decoder_input_dim * decoder_out_dim)
            # self.hnet_bias = nn.Linear(HN_input_dim, decoder_out_dim)

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

        self.dropout = nn.Dropout(p=0.1)

        if self.model_args.CONTEXT_DROPOUT:
            self.context_dropout = nn.Dropout(p=0.1)

        if self.model_args.USE_NODE_DEPTH:
            self.node_depth_embed = nn.Linear(self.model_args.MAX_NODE_DEPTH, self.model_args.CONTEXT_EMBED_SIZE)

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
            print (initrange)
            self.context_embed_HN.weight.data.uniform_(-initrange, initrange)

            # initialize the hypernet following Jake's paper
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.hnet_embed_weight.weight.data.zero_()
            self.hnet_embed_weight.bias.data.uniform_(-initrange, initrange)
            self.hnet_embed_bias.weight.data.zero_()
            self.hnet_embed_bias.bias.data.zero_()

            initrange = cfg.MODEL.TRANSFORMER.DECODER_INIT
            if self.model_args.HN_DECODER_LAST_LAYER:
                self.hnet_decoder_weight.weight.data.zero_()
                self.hnet_decoder_weight.bias.data.uniform_(-initrange, initrange)
                self.hnet_decoder_bias.weight.data.zero_()
                self.hnet_decoder_bias.bias.data.zero_()
            else:
                for i in range(len(self.hnet_decoder_weight)):
                    self.hnet_decoder_weight[i].weight.data.zero_()
                    self.hnet_decoder_weight[i].bias.data.uniform_(-initrange, initrange)
                    self.hnet_decoder_bias[i].weight.data.zero_()
                    self.hnet_decoder_bias[i].bias.data.zero_()

        if self.model_args.CONTEXT_PE:
            initrange = cfg.MODEL.TRANSFORMER.EMBED_INIT
            self.context_embed_PE.weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, dropout_mask=None, unimal_ids=None):
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
            context_embedding_attention = self.context_embed_attention(obs_context)
            # context_embedding_attention *= math.sqrt(self.model_args.CONTEXT_EMBED_SIZE)
            # if self.model_args.USE_NODE_DEPTH:
            #     context_embedding_attention = context_embedding_attention + self.node_depth_embed(morphology_info['node_depth'])
            # if self.model_args.CONTEXT_DROPOUT:
            #     context_embedding_attention = self.context_dropout(context_embedding_attention)

            if self.model_args.RNN_CONTEXT:
                mask = torch.cat([morphology_info['node_path_mask'] for _ in range(self.model_args.NHEAD)], 0)
                context_embedding_attention = self.rnn_context_encoder_FA(
                    context_embedding_attention, 
                    mask=mask, 
                )

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
            if not self.model_args.DEPTH_INPUT_HN:
                context_embedding_HN = self.context_embed_HN(obs_context)
                # context_embedding_HN = self.context_encoder_HN(context_embedding_HN, src_key_padding_mask=obs_mask)
                context_embedding_HN = self.context_encoder_HN(context_embedding_HN)
            else:
                context_embedding_HN = morphology_info['node_depth']

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
            # self.HN_embed_weight = embed_weight
            # can't scale when using HN; otherwise cause instability during learning
            # obs_embed = obs_embed * math.sqrt(self.d_model)
        else:
            if self.model_args.PER_NODE_EMBED:
                obs_embed = (obs[:, :, :, None] * self.limb_embed_weights[:, unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_embed_bias[:, unimal_ids, :]
            else:
                obs_embed = self.limb_embed(obs)
                # obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        
        if self.model_args.EMBEDDING_SCALE:
            obs_embed *= math.sqrt(self.d_model)

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
        if self.model_args.POS_EMBEDDING in ["learnt", "abs"] and self.model_args.PE_POSITION == 'base':
            obs_embed = self.pos_embedding(obs_embed)
        if self.model_args.CONTEXT_PE:
            obs_embed = obs_embed + context_embedding_PE
            # obs_embed = self.compress_embedding_PE(torch.cat([obs_embed, context_embedding_PE], -1))
        if self.model_args.USE_SWAT_PE:
            obs_embed = self.swat_PE_encoder(obs_embed, morphology_info['traversals'])
        if self.model_args.USE_SEPARATE_PE:
            obs_embed = self.separate_PE_encoder(obs_embed, unimal_ids)

        # print (f'embedding after PE: mean: {obs_embed.detach().cpu().numpy().ravel().mean():.2f}, std:{obs_embed.detach().cpu().numpy().ravel().std():.2f}')
        # plt.hist(obs_embed.detach().cpu().numpy().ravel(), bins=100, range=(-5., 5.), histtype='step', label='after PE')
        # plt.legend()
        # plt.savefig(f'figures/graph_PE_check/{self.count}.png')
        # plt.close()
        # self.count += 1

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
        
        if self.model_args.USE_CONNECTIVITY_IN_ATTENTION:
            attn_mask = torch.cat([morphology_info['connectivity'][:, :, :, i] for i in range(4)], 0)
            src_key_padding_mask = None
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
            if self.model_args.HN_DECODER_LAST_LAYER:
                output = self.shared_decoder_layers(decoder_input)
                layer_w = self.hnet_decoder_weight(context_embedding_HN).reshape(self.seq_len, batch_size, self.decoder_dims[-2], self.decoder_dims[-1])
                layer_b = self.hnet_decoder_bias(context_embedding_HN)
                output = (output[:, :, :, None] * layer_w).sum(dim=-2, keepdim=False) + layer_b
            else:
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

        return output, attention_maps, dropout_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0., batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if batch_first:
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

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

    def forward(self, obs, act=None, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):
        
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
        if cfg.MODEL.TRANSFORMER.USE_CONNECTIVITY_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity'].bool()
        if cfg.MODEL.TRANSFORMER.USE_MORPHOLOGY_INFO_IN_ATTENTION:
            morphology_info['connectivity'] = obs_dict['connectivity']
        if cfg.MODEL.TRANSFORMER.USE_NODE_DEPTH:
            # (batch_size, seq_len, max_node_depth) ->(seq_len, batch_size, max_node_depth)
            morphology_info['node_depth'] = obs_dict['node_depth'].permute(1, 0, 2)
        if cfg.MODEL.TRANSFORMER.USE_SWAT_PE:
            # (batch_size, seq_len, traversal_num) ->(seq_len, batch_size, traversal_num)
            morphology_info['traversals'] = obs_dict['traversals'].permute(1, 0, 2).long()
        if cfg.MODEL.TRANSFORMER.RNN_CONTEXT:
            morphology_info['node_path_length'] = obs_dict['node_path_length']
            morphology_info['node_path_mask'] = obs_dict['node_path_mask'].bool()
        
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
        if compute_val:
            # Per limb critic values
            limb_vals, v_attention_maps, dropout_mask_v = self.v_net(
                obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
                return_attention=return_attention, dropout_mask=dropout_mask_v, 
                unimal_ids=unimal_ids, 
            )
            # Zero out mask values
            limb_vals = limb_vals * (1 - obs_mask.int())
            # Use avg/max to keep the magnitidue same instead of sum
            num_limbs = self.seq_len - torch.sum(obs_mask.int(), dim=1, keepdim=True)
            val = torch.divide(torch.sum(limb_vals, dim=1, keepdim=True), num_limbs)
        else:
            val, v_attention_maps, dropout_mask_v = 0., None, 0.

        mu, mu_attention_maps, dropout_mask_mu = self.mu_net(
            obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, 
            return_attention=return_attention, dropout_mask=dropout_mask_mu, 
            unimal_ids=unimal_ids, 
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
    def act(self, obs, return_attention=False, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None, compute_val=True):
        val, pi, v_attention_maps, mu_attention_maps, dropout_mask_v, dropout_mask_mu = self.ac(obs, return_attention=return_attention, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids, compute_val=compute_val)
        if not cfg.DETERMINISTIC:
            act = pi.sample()
        else:
            act = pi.loc
        logp = pi.log_prob(act)
        act_mask = obs["act_padding_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        self.v_attention_maps = v_attention_maps
        self.mu_attention_maps = mu_attention_maps
        return val, act, logp, dropout_mask_v, dropout_mask_mu
        # return val, act, logp, dropout_mask_v, dropout_mask_mu, v_attention_maps, mu_attention_maps

    @torch.no_grad()
    def get_value(self, obs, dropout_mask_v=None, dropout_mask_mu=None, unimal_ids=None):
        val, _, _, _, _, _ = self.ac(obs, dropout_mask_v=dropout_mask_v, dropout_mask_mu=dropout_mask_mu, unimal_ids=unimal_ids)
        return val
