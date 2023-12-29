import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from metamorph.config import cfg
from metamorph.utils import model as tu

from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual


class MLPObsEncoder(nn.Module):
    """Encoder for env obs like hfield."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS + [cfg.MODEL.MLP.HIDDEN_DIM]
        self.encoder = tu.make_mlp_default(mlp_dims, final_nonlinearity=False)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)


class MLPModel(nn.Module):
    def __init__(self, obs_space, out_dim):
        super(MLPModel, self).__init__()
        self.model_args = cfg.MODEL.MLP
        self.seq_len = cfg.MODEL.MAX_LIMBS
        self.limb_obs_size = limb_obs_size = obs_space["proprioceptive"].shape[0] // self.seq_len
        self.limb_out_dim = out_dim // self.seq_len

        if self.model_args.ONE_HOT_CONTEXT:
            context_obs_size = cfg.MODEL.MAX_LIMBS
        else:
            context_obs_size = obs_space["context"].shape[0] // self.seq_len
        if self.model_args.LIMB_ONE_HOT:
            context_encoder_dim = [context_obs_size + self.seq_len] + [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE for _ in range(cfg.MODEL.TRANSFORMER.HN_CONTEXT_LAYER_NUM)]
        else:
            context_encoder_dim = [context_obs_size] + [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE for _ in range(cfg.MODEL.TRANSFORMER.HN_CONTEXT_LAYER_NUM)]
        HN_input_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE

        # set the input and output layer
        if self.model_args.HN_INPUT:
            print ('use HN for input layer')
            if self.model_args.CONTEXT_ENCODER_TYPE == 'linear':
                self.context_encoder_for_input = tu.make_mlp_default(context_encoder_dim)
            elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                context_embed = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                context_encoder_layers = TransformerEncoderLayerResidual(
                    cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE,
                    cfg.MODEL.TRANSFORMER.NHEAD,
                    256,
                    cfg.MODEL.TRANSFORMER.DROPOUT, 
                    batch_first=True, 
                )
                context_encoder_TF = TransformerEncoder(
                    context_encoder_layers, 1, norm=None,
                )
                if self.model_args.CONTEXT_MASK:
                    self.context_embed_input = context_embed
                    self.context_encoder_for_input = context_encoder_TF
                else:
                    self.context_encoder_for_input = nn.Sequential(
                        context_embed, 
                        context_encoder_TF, 
                    )
            elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                # self.context_embed_input = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                self.context_encoder_for_input = GraphNeuralNetwork(
                    input_dim = context_obs_size, 
                    hidden_dims = [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE], 
                    output_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE, 
                    final_nonlinearity=True
                )

            self.hnet_input_weight = nn.Linear(HN_input_dim, limb_obs_size * self.model_args.HIDDEN_DIM, bias=self.model_args.BIAS_IN_HN_OUTPUT_LAYER)

            if self.model_args.HN_INIT_STRATEGY == 'p2_norm':
                # TODO: how about changing initrange to the same way as a vanilla MLP in the base net?
                initrange = np.sqrt(1 / (self.limb_obs_size * self.seq_len))
                self.hnet_input_weight.weight.data.normal_(std=initrange)
                if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                    self.hnet_input_weight.bias.data.zero_()
            elif self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                # Var = d_input * d_context * limb_num * E[e^2]
                if self.model_args.HN_GENERATE_BIAS:
                    scale = 2.
                else:
                    scale = 1.
                var = 2. / (scale * (self.limb_obs_size * self.seq_len * HN_input_dim))
                initrange = np.sqrt(var)
                self.hnet_input_weight.weight.data.normal_(std=initrange)
                self.hnet_input_weight.bias.data.zero_()
            elif self.model_args.HN_INIT_STRATEGY == 'bias_init':
                initrange = np.sqrt(1 / (self.limb_obs_size * self.seq_len))
                self.hnet_input_weight.weight.data.zero_()
                self.hnet_input_weight.bias.data.normal_(std=initrange)
            elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                initrange = np.sqrt(1 / (self.limb_obs_size * self.seq_len))
                self.hnet_input_weight.weight.data.zero_()
                self.hnet_input_weight.bias.data.uniform_(-initrange, initrange)
            else:
                # use a heuristic value as the init range
                initrange = 0.001
                self.hnet_input_weight.weight.data.uniform_(-initrange, initrange)
                if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                    self.hnet_input_weight.bias.data.zero_()

            if self.model_args.HN_GENERATE_BIAS:
                self.hnet_input_bias = nn.Linear(HN_input_dim, self.model_args.HIDDEN_DIM)
                if self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                    var = 1. / HN_input_dim
                    initrange = np.sqrt(var)
                    self.hnet_input_bias.weight.data.normal_(std=initrange)
                    self.hnet_input_bias.bias.data.zero_()
                elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                    initrange = np.sqrt(1 / (self.limb_obs_size * self.seq_len))
                    self.hnet_input_bias.weight.data.zero_()
                    self.hnet_input_bias.bias.data.uniform_(-initrange, initrange)
                else:
                    self.hnet_input_bias.weight.data.zero_()
                    self.hnet_input_bias.bias.data.zero_()
            else:
                self.hidden_bias = nn.Parameter(torch.zeros(1, self.model_args.HIDDEN_DIM))

        elif self.model_args.PER_NODE_EMBED:
            print ('independent input weights for each node')
            initrange = 0.04
            self.limb_embed_weights = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))
            self.limb_embed_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM).uniform_(-initrange, initrange))

        else:
            self.input_layer = nn.Linear(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM)
            initrange = np.sqrt(1 / obs_space["proprioceptive"].shape[0])
            self.input_layer.weight.data.normal_(std=initrange)
            self.input_layer.bias.data.zero_()

        if self.model_args.HN_OUTPUT:
            print ('use HN for output layer')
            if not self.model_args.SHARE_CONTEXT_ENCODER:
                if self.model_args.CONTEXT_ENCODER_TYPE == 'linear':
                    self.context_encoder_for_output = tu.make_mlp_default(context_encoder_dim)
                elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                    context_embed = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                    context_encoder_layers = TransformerEncoderLayerResidual(
                        cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE,
                        cfg.MODEL.TRANSFORMER.NHEAD,
                        256,
                        cfg.MODEL.TRANSFORMER.DROPOUT,
                        batch_first=True, 
                    )
                    context_encoder_TF = TransformerEncoder(
                        context_encoder_layers, 1, norm=None,
                    )
                    if self.model_args.CONTEXT_MASK:
                        self.context_embed_output = context_embed
                        self.context_encoder_for_output = context_encoder_TF
                    else:
                        self.context_encoder_for_output = nn.Sequential(
                            context_embed, 
                            context_encoder_TF, 
                        )
                elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                    # self.context_embed_output = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                    self.context_encoder_for_output = GraphNeuralNetwork(
                        input_dim = context_obs_size, 
                        hidden_dims = [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE], 
                        output_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE, 
                        final_nonlinearity=True
                    )

            self.hnet_output_weight = nn.Linear(HN_input_dim, self.model_args.HIDDEN_DIM * self.limb_out_dim, bias=self.model_args.BIAS_IN_HN_OUTPUT_LAYER)

            if self.model_args.HN_INIT_STRATEGY == 'p2_norm':
                initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                self.hnet_output_weight.weight.data.normal_(std=initrange)
                if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                    self.hnet_output_weight.bias.data.zero_()
            elif self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                # Var = d_hidden * d_contexts * E[e^2]
                if self.model_args.HN_GENERATE_BIAS:
                    scale = 2.
                else:
                    scale = 1.
                var = 1. / (scale * (self.model_args.HIDDEN_DIM * HN_input_dim))
                initrange = np.sqrt(var)
                self.hnet_output_weight.weight.data.normal_(std=initrange)
                self.hnet_output_weight.bias.data.zero_()
            elif self.model_args.HN_INIT_STRATEGY == 'bias_init':
                initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                self.hnet_output_weight.weight.data.zero_()
                self.hnet_output_weight.bias.data.normal_(std=initrange)
            elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                self.hnet_output_weight.weight.data.zero_()
                self.hnet_output_weight.bias.data.uniform_(-initrange, initrange)
            else:
                initrange = 0.001
                self.hnet_output_weight.weight.data.uniform_(-initrange, initrange)
                if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                    self.hnet_output_weight.bias.data.zero_()

            if self.model_args.HN_GENERATE_BIAS:
                self.hnet_output_bias = nn.Linear(HN_input_dim, self.limb_out_dim)
                if self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                    var = 1. / (2 * HN_input_dim)
                    initrange = np.sqrt(var)
                    self.hnet_output_bias.weight.data.normal_(std=initrange)
                    self.hnet_output_bias.bias.data.zero_()
                elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                    initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                    self.hnet_output_bias.weight.data.zero_()
                    self.hnet_output_bias.bias.data.uniform_(-initrange, initrange)
                else:
                    self.hnet_output_bias.weight.data.zero_()
                    self.hnet_output_bias.bias.data.zero_()

        elif self.model_args.PER_NODE_DECODER:
            print ('independent output weights for each node')
            initrange = 1. / 16
            self.limb_output_weights = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), self.model_args.HIDDEN_DIM, out_dim).uniform_(-initrange, initrange))
            self.limb_output_bias = nn.Parameter(torch.zeros(len(cfg.ENV.WALKERS), out_dim).uniform_(-initrange, initrange))

        else:
            self.output_layer = nn.Linear(self.model_args.HIDDEN_DIM, out_dim, bias=False)
            initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
            self.output_layer.weight.data.normal_(std=initrange)

        if self.model_args.LAYER_NUM > 1:
            if self.model_args.HN_HIDDEN:
                print ('use HN for hidden layer')
                if not self.model_args.SHARE_CONTEXT_ENCODER:
                    if self.model_args.CONTEXT_ENCODER_TYPE == 'linear':
                        self.context_encoder_for_hidden = tu.make_mlp_default(context_encoder_dim)
                    elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                        context_embed = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                        context_encoder_layers = TransformerEncoderLayerResidual(
                            cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE,
                            cfg.MODEL.TRANSFORMER.NHEAD,
                            256,
                            cfg.MODEL.TRANSFORMER.DROPOUT,
                            batch_first=True, 
                        )
                        context_encoder_TF = TransformerEncoder(
                            context_encoder_layers, 1, norm=None,
                        )
                        if self.model_args.CONTEXT_MASK:
                            self.context_embed_hidden = context_embed
                            self.context_encoder_for_hidden = context_encoder_TF
                        else:
                            self.context_encoder_for_hidden = nn.Sequential(
                                context_embed, 
                                context_encoder_TF, 
                            )
                    elif self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                        # self.context_embed_output = nn.Linear(context_obs_size, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE)
                        self.context_encoder_for_hidden = GraphNeuralNetwork(
                            input_dim = context_obs_size, 
                            hidden_dims = [cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE], 
                            output_dim = cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE, 
                            final_nonlinearity=True
                        )

                HN_output_layers = []
                self.hidden_dims = [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM)]
                for i in range(self.model_args.LAYER_NUM - 1):
                    output_layer = nn.Linear(HN_input_dim, self.hidden_dims[i] * self.hidden_dims[i + 1], bias=self.model_args.BIAS_IN_HN_OUTPUT_LAYER)
                    if self.model_args.HN_INIT_STRATEGY == 'p2_norm':
                        initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                        output_layer.weight.data.normal_(std=initrange)
                        if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                            output_layer.bias.data.zero_()
                    elif self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                        # Var = d_hidden * d_contexts * E[e^2]
                        if self.model_args.HN_GENERATE_BIAS:
                            scale = 2.
                        else:
                            scale = 1.
                        var = 1. / (scale * (self.model_args.HIDDEN_DIM * HN_input_dim))
                        initrange = np.sqrt(var)
                        output_layer.weight.data.normal_(std=initrange)
                        output_layer.bias.data.zero_()
                    elif self.model_args.HN_INIT_STRATEGY == 'bias_init':
                        initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                        output_layer.weight.data.zero_()
                        output_layer.bias.data.normal_(std=initrange)
                    elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                        initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                        output_layer.weight.data.zero_()
                        output_layer.bias.data.uniform_(-initrange, initrange)
                    else:
                        initrange = 0.001
                        output_layer.weight.data.uniform_(-initrange, initrange)
                        if self.model_args.BIAS_IN_HN_OUTPUT_LAYER:
                            output_layer.bias.data.zero_()
                    HN_output_layers.append(output_layer)
                self.HN_hidden_weight = nn.ModuleList(HN_output_layers)

                if self.model_args.HN_GENERATE_BIAS:
                    layers = []
                    for i in range(self.model_args.LAYER_NUM - 1):
                        hnet_hidden_bias = nn.Linear(HN_input_dim, self.model_args.HIDDEN_DIM)
                        if self.model_args.HN_INIT_STRATEGY == 'HN_paper':
                            var = 1. / HN_input_dim
                            initrange = np.sqrt(var)
                            hnet_hidden_bias.weight.data.normal_(std=initrange)
                            hnet_hidden_bias.bias.data.zero_()
                        elif self.model_args.HN_INIT_STRATEGY == 'bias_init_v2':
                            initrange = np.sqrt(1 / self.model_args.HIDDEN_DIM)
                            hnet_hidden_bias.weight.data.zero_()
                            hnet_hidden_bias.bias.data.uniform_(-initrange, initrange)
                        else:
                            hnet_hidden_bias.weight.data.zero_()
                            hnet_hidden_bias.bias.data.zero_()
                        layers.append(hnet_hidden_bias)
                    self.HN_hidden_bias = nn.ModuleList(layers)

        if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])

        if self.model_args.LAYER_NORM:
            norm_layers = [nn.LayerNorm(cfg.MODEL.MLP.HIDDEN_DIM, elementwise_affine=False) for _ in range(cfg.MODEL.MLP.LAYER_NUM)]
            self.LN_layers = nn.ModuleList(norm_layers)

        if self.model_args.CONTEXT_EMBEDDING_NORM == 'layer_norm':
            self.layer_norm_input = nn.LayerNorm(HN_input_dim)
            self.layer_norm_hidden = nn.LayerNorm(HN_input_dim)
            self.layer_norm_output = nn.LayerNorm(HN_input_dim)

        if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
            self.input_dropout = nn.Dropout(p=0.1)
            self.hidden_dropout = nn.Dropout(p=0.1)
            self.output_dropout = nn.Dropout(p=0.1)


    def forward(self, obs, obs_mask, obs_env, obs_cm_mask, obs_context, morphology_info, return_attention=False, unimal_ids=None):

        batch_size = obs.shape[0]

        if not self.training:
            obs = obs.view(batch_size, self.seq_len, -1)
            embedding = (obs[:, :, :, None] * self.input_weight).sum(dim=-2) + self.input_bias
            embedding = embedding * (1. - obs_mask.float())[:, :, None]
            # aggregate all limbs' embedding
            if cfg.MODEL.MLP.RELU_BEFORE_AGG:
                embedding = F.relu(embedding)
            if cfg.MODEL.MLP.INPUT_AGGREGATION == 'limb_num':
                embedding = embedding.sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
            elif cfg.MODEL.MLP.INPUT_AGGREGATION == 'sqrt_limb_num':
                embedding = embedding.sum(dim=1) / torch.sqrt((1. - obs_mask.float()).sum(dim=1, keepdim=True))
            elif cfg.MODEL.MLP.INPUT_AGGREGATION == 'max_limb_num':
                embedding = embedding.mean(dim=1)
            else:
                embedding = embedding.sum(dim=1)

            if "hfield" in cfg.ENV.KEYS_TO_KEEP:
                hfield_embedding = self.hfield_encoder(obs_env["hfield"])
                embedding = embedding + hfield_embedding

            embedding = F.relu(embedding)

            # if "hfield" in cfg.ENV.KEYS_TO_KEEP:
            #     hfield_embedding = self.hfield_encoder(obs_env["hfield"])
            #     embedding = torch.cat([embedding, hfield_embedding], 1)

            for weight, bias in zip(self.hidden_weights, self.hidden_bias):
                embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
                embedding = F.relu(embedding)

            output = (embedding[:, None, :, None] * self.output_weight).sum(dim=-2) + self.output_bias
            output = output.reshape(batch_size, -1)
            return output, None

        obs_context = obs_context.view(batch_size, self.seq_len, -1)
        if self.model_args.LIMB_ONE_HOT:
            # concatenate one-hot limb id to the context features
            coding = torch.eye(self.seq_len, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)
            obs_context = torch.cat([obs_context, coding], dim=-1)

        # input layer
        if self.model_args.HN_INPUT:

            if self.model_args.ONE_HOT_CONTEXT:
                context_embedding = torch.eye(self.seq_len, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                context_embedding = obs_context

            if self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                context_embedding = self.context_encoder_for_input(context_embedding, morphology_info["adjacency_matrix"])
            elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                if self.model_args.CONTEXT_MASK:
                    context_embedding = self.context_embed_input(context_embedding)
                    context_embedding = self.context_encoder_for_input(context_embedding, src_key_padding_mask=obs_mask)
                else:
                    context_embedding = self.context_encoder_for_input(context_embedding)
            else:
                context_embedding = self.context_encoder_for_input(context_embedding)
            if self.model_args.HN_INIT_STRATEGY == 'p2_norm':
                # TODO: should we include gradient for the normalization op?
                context_embedding = torch.div(context_embedding, torch.norm(context_embedding, p=2, dim=-1, keepdim=True))
            if self.model_args.CONTEXT_EMBEDDING_NORM == 'p2_norm':
                context_embedding = torch.div(context_embedding, torch.norm(context_embedding, p=2, dim=-1, keepdim=True))
            elif self.model_args.CONTEXT_EMBEDDING_NORM == 'layer_norm':
                context_embedding = self.layer_norm_input(context_embedding)
            if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                context_embedding = self.input_dropout(context_embedding)
            input_weight = self.hnet_input_weight(context_embedding).view(batch_size, self.seq_len, self.limb_obs_size, self.model_args.HIDDEN_DIM)
            # save for diagnose
            self.context_embedding_input = context_embedding
            self.input_weight = input_weight

            obs = obs.view(batch_size, self.seq_len, -1)
            if self.model_args.HN_GENERATE_BIAS:
                input_bias = self.hnet_input_bias(context_embedding)
                embedding = (obs[:, :, :, None] * input_weight).sum(dim=-2) + input_bias
            else:
                embedding = (obs[:, :, :, None] * input_weight).sum(dim=-2)
            # setting zero-padding limbs' values to 0
            # embedding shape: batch_size * limb_num * hidden_layer_dim
            embedding = embedding * (1. - obs_mask.float())[:, :, None]

            # aggregate all limbs' embedding
            if cfg.MODEL.MLP.RELU_BEFORE_AGG:
                embedding = F.relu(embedding)
            if cfg.MODEL.MLP.INPUT_AGGREGATION == 'limb_num':
                embedding = embedding.sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
            elif cfg.MODEL.MLP.INPUT_AGGREGATION == 'sqrt_limb_num':
                embedding = embedding.sum(dim=1) / torch.sqrt((1. - obs_mask.float()).sum(dim=1, keepdim=True))
            elif cfg.MODEL.MLP.INPUT_AGGREGATION == 'max_limb_num':
                embedding = embedding.mean(dim=1)
            else:
                embedding = embedding.sum(dim=1)
            # add bias
            if not self.model_args.HN_GENERATE_BIAS:
                embedding = embedding + self.hidden_bias
            if self.model_args.LAYER_NORM:
                embedding = self.LN_layers[0](embedding)

            if "hfield" in cfg.ENV.KEYS_TO_KEEP:
                hfield_embedding = self.hfield_encoder(obs_env["hfield"])
                embedding = embedding + hfield_embedding

            embedding = F.relu(embedding)
        
        # hidden layers
        if self.model_args.LAYER_NUM > 1:
            if self.model_args.HN_HIDDEN:
                if self.model_args.SHARE_CONTEXT_ENCODER:
                    context_embedding = self.context_embedding_input
                else:
                    context_embedding = obs_context
                    if self.model_args.CONTEXT_MASK:
                        context_embedding = self.context_embed_hidden(context_embedding)
                        context_embedding = self.context_encoder_for_hidden(context_embedding, src_key_padding_mask=obs_mask)
                    else:
                        context_embedding = self.context_encoder_for_hidden(context_embedding)
                    # aggregate the context embedding
                    # need to aggregate with mean. sum will lead to significant KL divergence
                    context_embedding = (context_embedding * (1. - obs_mask.float())[:, :, None]).sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
                    if self.model_args.CONTEXT_EMBEDDING_NORM == 'p2_norm':
                        context_embedding = torch.div(context_embedding, torch.norm(context_embedding, p=2, dim=-1, keepdim=True))
                    elif self.model_args.CONTEXT_EMBEDDING_NORM == 'layer_norm':
                        context_embedding = self.layer_norm_hidden(context_embedding)
                    if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                        context_embedding = self.hidden_dropout(context_embedding)
                    for i, layer in enumerate(self.HN_hidden_weight):
                        weight = layer(context_embedding).view(batch_size, self.hidden_dims[i], self.hidden_dims[i + 1])
                        if self.model_args.HN_GENERATE_BIAS:
                            bias = self.HN_hidden_bias[i](context_embedding)
                            embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
                        else:
                            embedding = (embedding[:, :, None] * weight).sum(dim=1)
                        if self.model_args.LAYER_NORM:
                            embedding = self.LN_layers[i + 1](embedding)
                        embedding = F.relu(embedding)
            else:
                embedding = self.hidden_layers(embedding)

        # output layer
        if self.model_args.HN_OUTPUT:

            if self.model_args.ONE_HOT_CONTEXT:
                context_embedding = torch.eye(self.seq_len, device='cuda').unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                context_embedding = obs_context
            
            if self.model_args.SHARE_CONTEXT_ENCODER:
                context_embedding = self.context_embedding_input
            else:
                if self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                    context_embedding = self.context_encoder_for_output(context_embedding, morphology_info["adjacency_matrix"])
                elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                    if self.model_args.CONTEXT_MASK:
                        context_embedding = self.context_embed_output(context_embedding)
                        context_embedding = self.context_encoder_for_output(context_embedding, src_key_padding_mask=obs_mask)
                    else:
                        context_embedding = self.context_encoder_for_output(context_embedding)
                else:
                    context_embedding = self.context_encoder_for_output(context_embedding)
                if self.model_args.HN_INIT_STRATEGY == 'p2_norm':
                    context_embedding = torch.div(context_embedding, torch.norm(context_embedding, p=2, dim=-1, keepdim=True))
            if self.model_args.CONTEXT_EMBEDDING_NORM == 'p2_norm':
                context_embedding = torch.div(context_embedding, torch.norm(context_embedding, p=2, dim=-1, keepdim=True))
            elif self.model_args.CONTEXT_EMBEDDING_NORM == 'layer_norm':
                context_embedding = self.layer_norm_output(context_embedding)
            if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                context_embedding = self.output_dropout(context_embedding)
            output_weight = self.hnet_output_weight(context_embedding).view(batch_size, self.seq_len, self.model_args.HIDDEN_DIM, self.limb_out_dim)
            # save for diagnose
            self.context_embedding_output = context_embedding
            self.output_weight = output_weight

            if self.model_args.HN_GENERATE_BIAS:
                output_bias = self.hnet_output_bias(context_embedding)
                output = (embedding[:, None, :, None] * output_weight).sum(dim=-2) + output_bias
            else:
                output = (embedding[:, None, :, None] * output_weight).sum(dim=-2)
            output = output.reshape(batch_size, -1)

        elif self.model_args.PER_NODE_DECODER:
            output = (embedding[:, :, None] * self.limb_output_weights[unimal_ids, :, :]).sum(dim=-2, keepdim=False) + self.limb_output_bias[unimal_ids]

        else:
            output = self.output_layer(embedding)

        return output, None

    @torch.no_grad()
    def generate_params(self, obs_context, obs_mask, morphology_info=None):

        batch_size = obs_context.shape[0]
        obs_context = obs_context.view(batch_size, self.seq_len, -1)

        # input layer
        if self.model_args.HN_INPUT:
            context_embedding = obs_context
            if self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                context_embedding = self.context_encoder_for_input(context_embedding, morphology_info["adjacency_matrix"])
            elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                if self.model_args.CONTEXT_MASK:
                    context_embedding = self.context_embed_input(context_embedding)
                    context_embedding = self.context_encoder_for_input(context_embedding, src_key_padding_mask=obs_mask)
                else:
                    context_embedding = self.context_encoder_for_input(context_embedding)
            else:
                context_embedding = self.context_encoder_for_input(context_embedding)
            if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                context_embedding = self.input_dropout(context_embedding)
            self.input_weight = self.hnet_input_weight(context_embedding).view(batch_size, self.seq_len, self.limb_obs_size, self.model_args.HIDDEN_DIM)
            self.input_bias = self.hnet_input_bias(context_embedding)

        # hidden layers
        if self.model_args.LAYER_NUM > 1:
            if self.model_args.HN_HIDDEN:
                if self.model_args.SHARE_CONTEXT_ENCODER:
                    context_embedding = self.context_embedding_input
                else:
                    context_embedding = obs_context
                    if self.model_args.CONTEXT_MASK:
                        context_embedding = self.context_embed_hidden(context_embedding)
                        context_embedding = self.context_encoder_for_hidden(context_embedding, src_key_padding_mask=obs_mask)
                    else:
                        context_embedding = self.context_encoder_for_hidden(context_embedding)
                    # aggregate the context embedding
                    # need to aggregate with mean. sum will lead to significant KL divergence
                    context_embedding = (context_embedding * (1. - obs_mask.float())[:, :, None]).sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
                    if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                        context_embedding = self.hidden_dropout(context_embedding)
                    self.hidden_weights, self.hidden_bias = [], []
                    for i, layer in enumerate(self.HN_hidden_weight):
                        weight = layer(context_embedding).view(batch_size, self.hidden_dims[i], self.hidden_dims[i + 1])
                        self.hidden_weights.append(weight)
                        bias = self.HN_hidden_bias[i](context_embedding)
                        self.hidden_bias.append(bias)

        # output layer
        if self.model_args.HN_OUTPUT:
            
            if self.model_args.SHARE_CONTEXT_ENCODER:
                context_embedding = self.context_embedding_input
            else:
                context_embedding = obs_context
                if self.model_args.CONTEXT_ENCODER_TYPE == 'gnn':
                    context_embedding = self.context_encoder_for_output(context_embedding, morphology_info["adjacency_matrix"])
                elif self.model_args.CONTEXT_ENCODER_TYPE == 'transformer':
                    if self.model_args.CONTEXT_MASK:
                        context_embedding = self.context_embed_output(context_embedding)
                        context_embedding = self.context_encoder_for_output(context_embedding, src_key_padding_mask=obs_mask)
                    else:
                        context_embedding = self.context_encoder_for_output(context_embedding)
                else:
                    context_embedding = self.context_encoder_for_output(context_embedding)
            if self.model_args.CONTEXT_EMBEDDING_DROPOUT:
                context_embedding = self.output_dropout(context_embedding)
            self.output_weight = self.hnet_output_weight(context_embedding).view(batch_size, self.seq_len, self.model_args.HIDDEN_DIM, self.limb_out_dim)
            self.output_bias = self.hnet_output_bias(context_embedding)
