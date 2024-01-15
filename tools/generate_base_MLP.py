import argparse
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.inherit_weight import restore_from_checkpoint
from metamorph.algos.ppo.model import ActorCritic

from tools.train_ppo import set_cfg_options


class BaseMLP(nn.Module):
    def __init__(self):
        super(BaseMLP, self).__init__()
        pass

        # # input layer
        # self.input_layer = nn.Linear(obs_space["proprioceptive"].shape[0], self.model_args.HIDDEN_DIM)
        # # hidden layers
        # if self.model_args.LAYER_NUM > 1:
        #     hidden_dims = [self.model_args.HIDDEN_DIM for _ in range(self.model_args.LAYER_NUM)]
        #     if "hfield" in cfg.ENV.KEYS_TO_KEEP:
        #         hidden_dims[0] += cfg.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS[-1]
        #     self.hidden_layers = tu.make_mlp_default(hidden_dims)
        # # hfield
        # if "hfield" in cfg.ENV.KEYS_TO_KEEP:
        #     self.hfield_encoder = MLPObsEncoder(obs_space.spaces["hfield"].shape[0])
        # # output layer
        # self.final_input_dim = self.model_args.HIDDEN_DIM 
        # self.output_layer = nn.Linear(self.final_input_dim, out_dim)

    def copy_params(self, policy, idx):
        self.input_weight = policy.input_weight[idx].unsqueeze(0).clone()
        self.input_bias = policy.input_bias[idx].unsqueeze(0).clone()

        try:
            self.hfield_encoder = policy.hfield_encoder.clone()
        except:
            pass
        
        self.hidden_weights = [params[idx].unsqueeze(0).clone() for params in policy.hidden_weights]
        self.hidden_bias = [params[idx].unsqueeze(0).clone() for params in policy.hidden_bias]

        self.output_weight = policy.output_weight[idx].unsqueeze(0).clone()
        self.output_bias = policy.output_bias[idx].unsqueeze(0).clone()

    def forward(self, obs, obs_mask, obs_env):

        # obs should have shape batch_size * seq_len * feat_dim
        embedding = (obs[:, :, :, None] * self.input_weight).sum(dim=-2) + self.input_bias
        embedding = embedding * (1. - obs_mask.float())[:, :, None]
        # aggregate all limbs' embedding
        embedding = embedding.sum(dim=1) / (1. - obs_mask.float()).sum(dim=1, keepdim=True)
        embedding = F.relu(embedding)

        if "hfield" in obs_env:
            hfield_embedding = self.hfield_encoder(obs_env["hfield"])
            embedding = torch.cat([embedding, hfield_embedding], 1)

        for weight, bias in zip(self.hidden_weights, self.hidden_bias):
            embedding = (embedding[:, :, None] * weight).sum(dim=1) + bias
            embedding = F.relu(embedding)

        output = (embedding[:, None, :, None] * self.output_weight).sum(dim=-2) + self.output_bias
        return output


def copy_params(args):

    path = f'{args.folder}/{args.seed}'
    cfg.merge_from_file(f'{path}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'{path}/checkpoint_50.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = args.agent_path
    set_cfg_options()
    ppo_trainer = PPO()
    policy = ppo_trainer.agent.ac
    obs_rms = get_ob_rms(ppo_trainer.envs)
    # change to eval mode as we have dropout in the model
    policy.eval()

    agents = [x[:-4] for x in os.listdir(f'{args.agent_path}/xml')][:3]
    context, obs_mask, act_mask = [], [], []
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        init_obs = env.reset()
        env.close()
        context.append(init_obs['context'])
        obs_mask.append(init_obs['obs_padding_mask'])
    context = np.stack(context, axis=0)
    context = torch.from_numpy(context).float().cuda()
    obs_mask = np.stack(obs_mask, axis=0)
    obs_mask = torch.from_numpy(obs_mask).bool().cuda()

    # generate base weights
    policy.mu_net.generate_params(context, obs_mask)
    # copy weights to separate base MLPs
    save_path = '/'.join(['base_MLP'] + path.split('/')[1:])
    os.makedirs(save_path, exist_ok=True)
    for i, agent in enumerate(agents):
        net = BaseMLP()
        net.copy_params(policy.mu_net, i)
        torch.save(net, f'{save_path}/{agent}.pt')

    for i, agent in enumerate(agents):
        net = torch.load(f'{save_path}/{agent}.pt')
        with open(f'expert_data/ft_baseline_KL_5_wo_PE+dropout/{args.seed}/{agent}.pkl', 'rb') as f:
            agent_data = pickle.load(f)
        obs = agent_data['obs'][:100].reshape(100, cfg.MODEL.MAX_LIMBS, -1)
        dims = list(range(13)) + [30, 31] + [41, 42]
        obs = obs[:, :, dims].cuda()
        agent_context = context[i].unsqueeze(0).repeat(100, 1).cuda()
        agent_mask = obs_mask[i].unsqueeze(0).repeat(100, 1)
        obs_env = {}
        try:
            obs_env['hfield'] = agent_data['hfield'][:100].cuda()
        except:
            pass
        output = net(obs, agent_mask, obs_env)
        output = output.reshape(100, -1)
        policy.mu_net.generate_params(agent_context, agent_mask)
        output_true, _ = policy.mu_net(obs, agent_mask, obs_env, None, agent_context, None)
        print (output[2])
        print (output_true[2])
        print ((output != output_true).float().sum())


def compare_speed(folder, seed, agent_path, sample_num, model_type='HN'):
    path = f'{folder}/{seed}'
    cfg.merge_from_file(f'{path}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'{path}/checkpoint_50.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = args.agent_path
    set_cfg_options()
    ppo_trainer = PPO()
    policy = ppo_trainer.agent.ac
    # obs_rms = get_ob_rms(ppo_trainer.envs)
    # change to eval mode as we have dropout in the model
    policy.eval()

    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')][:10]
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        init_obs = env.reset()
        env.close()
        context = torch.from_numpy(init_obs['context']).float().unsqueeze(0).repeat(sample_num, 1).cuda()
        obs_mask = torch.from_numpy(init_obs['obs_padding_mask']).bool().unsqueeze(0).repeat(sample_num, 1).cuda()
        with open(f'expert_data/ft_baseline_KL_5_wo_PE+dropout/{seed}/{agent}.pkl', 'rb') as f:
            agent_data = pickle.load(f)
        obs = agent_data['obs'][:sample_num].cuda().reshape(100, cfg.MODEL.MAX_LIMBS, -1)
        if model_type == 'HN':
            policy.mu_net.generate_params(context, obs_mask)
            dims = list(range(13)) + [30, 31] + [41, 42]
            obs = obs[:, :, dims].reshape(sample_num, -1)
        else:
            obs = obs.permute(1, 0, 2)
        obs_env = {}
        try:
            obs_env['hfield'] = agent_data['hfield'][:sample_num].cuda()
        except:
            pass
        start = time.time()
        # policy.mu_net(obs, obs_mask, obs_env, None, context, None)
        if model_type == 'HN':
            for i in range(sample_num):
                policy.mu_net(obs[i].unsqueeze(0), obs_mask[i].unsqueeze(0), obs_env, None, context[i].unsqueeze(0), None)
        else:
            for i in range(sample_num):
                policy.mu_net(obs[:, i, :].unsqueeze(1), obs_mask[i].unsqueeze(0), obs_env, None, context[i].unsqueeze(0), None)
        end = time.time()
        duration = end - start
        print (agent, duration)

        simple_mlp = nn.Sequential(
            nn.Linear(17 * 12, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 24)
        ).cuda()
        start = time.time()
        for i in range(sample_num):
            simple_mlp(obs[i].unsqueeze(0))
        end = time.time()
        duration = end - start
        print ('true MLP', agent, duration)



if __name__ == '__main__':

    # python tools/generate_base_MLP.py --folder distilled_policy/ft_MT_TF_to_HN-MLP_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000
    # python tools/generate_base_MLP.py --folder distilled_policy/ft_MT_TF_to_TF_lr_3e-4_expert_size_64000
    parser = argparse.ArgumentParser(description="Generate base MLP with HN")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--agent_path", default='data/train', type=str)
    parser.add_argument("--seed", type=str, default='1409')
    args = parser.parse_args()
    # copy_params(args)
    if 'HN-MLP' in args.folder:
        model_type = 'HN'
    else:
        model_type = 'TF'
    sample_num = 100
    compare_speed(args.folder, args.seed, args.agent_path, sample_num, model_type=model_type)
