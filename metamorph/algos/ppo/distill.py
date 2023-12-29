import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.model import ActorCritic, Agent

from tools.train_ppo import set_cfg_options

torch.manual_seed(0)
np.random.seed(0)


class DistillationDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, index):
        obs = self.data['obs'][index]
        act = self.data['act'][index]
        act_mean = self.data['act_mean'][index]
        context = self.data['context'][index]
        obs_mask = self.data['obs_padding_mask'][index]
        act_mask = self.data['act_padding_mask'][index]
        if 'hfield' in self.data:
            hfield = self.data['hfield'][index]
        else:
            hfield = None
        return obs, act, act_mean, context, obs_mask, act_mask, hfield


def distill_policy(source_folder, target_folder, agents):

    cfg.ENV.WALKER_DIR = 'data/train'
    cfg.ENV.WALKERS = agents
    set_cfg_options()
    envs = make_vec_envs()
    model = ActorCritic(envs.observation_space, envs.action_space).cuda()

    with open(f'expert_data/{source_folder}/obs_rms.pkl', 'rb') as f:
        obs_rms = pickle.load(f)
    if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6:
        dims = list(range(13)) + [30, 31] + [41, 42]
        new_mean = obs_rms['proprioceptive'].mean.reshape(cfg.MODEL.MAX_LIMBS, -1)
        obs_rms['proprioceptive'].mean = new_mean[:, dims].ravel()
        new_var = obs_rms['proprioceptive'].var.reshape(cfg.MODEL.MAX_LIMBS, -1)
        obs_rms['proprioceptive'].var = new_var[:, dims].ravel()

    # merge the training data
    buffer = {
        'obs': [], 
        'act': [], 
        'act_mean': [], 
        'context': [], 
        'obs_padding_mask': [], 
        'act_padding_mask': [], 
    }
    if cfg.ENV.KEYS_TO_KEEP:
        buffer['hfield'] = []
    for i, agent in enumerate(agents):
        # if i == 5:
        #     break
        data_path = f'expert_data/{source_folder}/{agent}.pkl'
        if not os.path.exists(data_path):
            continue
        print (agent)
        with open(data_path, 'rb') as f:
            agent_data = pickle.load(f)
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        init_obs = env.reset()
        env.close()
        # drop context features from obs if needed
        if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6:
            dims = list(range(13)) + [30, 31] + [41, 42]
            data_size = agent_data['obs'].shape[0]
            new_obs = agent_data['obs'].view(data_size, cfg.MODEL.MAX_LIMBS, -1)
            new_obs = new_obs[:, :, dims]
            agent_data['obs'] = new_obs.view(data_size, -1)
        if cfg.DISTILL.SAMPLE_STRATEGY == 'random':
            sample_index = np.random.choice(agent_data['obs'].shape[0], cfg.DISTILL.PER_AGENT_SAMPLE_NUM, replace=False)
        for key in ['obs', 'act', 'act_mean']:
            if cfg.DISTILL.SAMPLE_STRATEGY == 'random':
                buffer[key].append(agent_data[key][sample_index])
            elif cfg.DISTILL.SAMPLE_STRATEGY == 'timestep_first':
                data_size = agent_data[key].shape[0]
                feat_dim = agent_data[key].shape[-1]
                buffer[key].append(agent_data[key].reshape(-1, 64, feat_dim).permute(1, 0, 2).reshape(data_size, feat_dim)[:cfg.DISTILL.PER_AGENT_SAMPLE_NUM])
            elif cfg.DISTILL.SAMPLE_STRATEGY == 'env_first':
                buffer[key].append(agent_data[key][:cfg.DISTILL.PER_AGENT_SAMPLE_NUM])
            else:
                raise ValueError("Unsupported sample strategy")
        for key in ['context', 'obs_padding_mask', 'act_padding_mask']:
            value = torch.from_numpy(init_obs[key]).float().unsqueeze(0).repeat(cfg.DISTILL.PER_AGENT_SAMPLE_NUM, 1)
            buffer[key].append(value)
        if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
            buffer['hfield'].append(agent_data['hfield'])

    for key in buffer:
        buffer[key] = torch.cat(buffer[key], dim=0)

    train_data = DistillationDataset(buffer)
    train_dataloader = DataLoader(train_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.DISTILL.BASE_LR, 
        eps=cfg.DISTILL.EPS, 
        weight_decay=cfg.DISTILL.WEIGHT_DECAY
    )

    for i in range(cfg.DISTILL.EPOCH_NUM):

        if i % cfg.DISTILL.SAVE_FREQ == 0:
            torch.save([model.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{i}.pt')

        batch_losses = []
        for obs, act, act_mean, context, obs_mask, act_mask, hfield in train_dataloader:

            obs_dict = {
                'proprioceptive': obs.cuda(), 
                'context': context.cuda(), 
                'obs_padding_mask': obs_mask.cuda(), 
                'act_padding_mask': act_mask.cuda(), 
            }
            if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                obs_dict['hfield'] = hfield.cuda()

            if cfg.DISTILL.IMITATION_TARGET == 'act':
                _, pi, logp, _ = model(obs_dict, act=act.cuda(), compute_val=False)
            else:
                _, pi, logp, _ = model(obs_dict, act=act_mean.cuda(), compute_val=False)
            if cfg.DISTILL.LOSS_TYPE == 'KL':
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = 0.5 * (((model.action_mu - act_mean.cuda()).square() * (1 - obs_dict['act_padding_mask'])).sum(dim=1) / (1 - obs_dict['act_padding_mask']).sum(dim=1)).mean()
                else:
                    loss = 0.5 * ((model.action_mu - act_mean.cuda()).square() * (1 - obs_dict['act_padding_mask'])).sum(dim=1).mean()
            elif cfg.DISTILL.LOSS_TYPE == 'logp':
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = -((model.limb_logp * (1 - obs_dict['act_padding_mask'])).sum(dim=1, keepdim=True) / (1 - obs_dict['act_padding_mask']).sum(dim=1, keepdim=True)).mean()
                else:
                    loss = -logp.mean()
            else:
                raise ValueError("Unsupported loss type")

            optimizer.zero_grad()
            loss.backward()

            if cfg.DISTILL.GRAD_NORM is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.DISTILL.GRAD_NORM)

            optimizer.step()

            batch_losses.append(loss.item())

        print (f'epoch {i}, average batch loss: {np.mean(batch_losses)}')

    torch.save([model.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{cfg.DISTILL.EPOCH_NUM}.pt')
