import numpy as np
import os
import pickle
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.model import ActorCritic, Agent
from metamorph.envs.vec_env.running_mean_std import RunningMeanStd

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
        unimal_ids = self.data['unimal_ids'][index]
        if 'hfield' in self.data:
            hfield = self.data['hfield'][index]
        else:
            hfield = torch.zeros(1)
        return obs, act, act_mean, hfield, unimal_ids


def distill_policy(source_folder, target_folder, teacher_mode, validation=False):

    agents = cfg.ENV.WALKERS
    envs = make_vec_envs()
    model = ActorCritic(envs.observation_space, envs.action_space).cuda()

    # merge the training data
    buffer = {
        'obs': [], 
        'act': [], 
        'act_mean': [], 
        'unimal_ids': [], 
    }
    if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
        buffer['hfield'] = []
    all_context = defaultdict(list)

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
        all_context['context'].append(init_obs['context'])
        all_context['obs_mask'].append(init_obs['obs_padding_mask'])
        all_context['act_mask'].append(init_obs['act_padding_mask'])
        all_context['adjacency_matrix'].append(init_obs['adjacency_matrix'])
        if type(agent_data['obs']) == list:
            # aggregate episode data
            episode_num = cfg.DISTILL.PER_AGENT_EPISODE_NUM
            for key in ['obs', 'act', 'act_mean', 'hfield']:
                if key not in agent_data:
                    continue
                agent_data[key] = torch.cat(agent_data[key][:episode_num], dim=0)
                if key == 'obs':
                    if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6 and agent_data['obs'].shape[-1] == 624:
                        dims = list(range(13)) + [30, 31] + [41, 42]
                        data_size = agent_data['obs'].shape[0]
                        new_obs = agent_data['obs'].view(data_size, cfg.MODEL.MAX_LIMBS, -1)
                        new_obs = new_obs[:, :, dims]
                        agent_data['obs'] = new_obs.view(data_size, -1)
                buffer[key].append(agent_data[key])
            buffer['unimal_ids'].append(torch.ones(agent_data['obs'].shape[0], dtype=torch.long) * i)
        else:
            # drop context features from obs if needed
            if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6 and agent_data['obs'].shape[-1] == 624:
                dims = list(range(13)) + [30, 31] + [41, 42]
                data_size = agent_data['obs'].shape[0]
                new_obs = agent_data['obs'].view(data_size, cfg.MODEL.MAX_LIMBS, -1)
                new_obs = new_obs[:, :, dims]
                agent_data['obs'] = new_obs.view(data_size, -1)
            if cfg.DISTILL.SAMPLE_STRATEGY == 'random':
                sample_index = np.random.choice(agent_data['obs'].shape[0], cfg.DISTILL.PER_AGENT_SAMPLE_NUM, replace=False)
            for key in ['obs', 'act', 'act_mean', 'hfield']:
                if key not in buffer:
                    continue
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
            buffer['unimal_ids'].append(torch.ones(cfg.DISTILL.PER_AGENT_SAMPLE_NUM, dtype=torch.long) * i)
    # all_context = np.stack(all_context, axis=0)
    # all_context = all_context.reshape(1200, -1)
    # for i in range(all_context.shape[1]):
    #     print (i, all_context[:, i].min(), all_context[:, i].max())
    # with open('context_norm.pkl', 'wb') as f:
    #     pickle.dump([all_context.min(axis=0), all_context.max(axis=0)], f)
    all_context['context'] = torch.from_numpy(np.stack(all_context['context'])).float().cuda()
    all_context['obs_mask'] = torch.from_numpy(np.stack(all_context['obs_mask'])).float().cuda()
    all_context['act_mask'] = torch.from_numpy(np.stack(all_context['act_mask'])).float().cuda()
    all_context['adjacency_matrix'] = torch.from_numpy(np.stack(all_context['adjacency_matrix'])).float().cuda()

    if validation:
        in_domain_validation_buffer = {}
        out_domain_validation_buffer = {}
    for key in buffer:
        if validation:
            valid_agent_num = int(len(buffer[key]) * 0.9)
            out_domain_validation_buffer[key] = torch.cat(buffer[key][valid_agent_num:], dim=0)
            valid_sample_num = int(cfg.DISTILL.PER_AGENT_SAMPLE_NUM * 0.9)
            in_domain_validation_buffer[key] = torch.cat([x[valid_sample_num:] for x in buffer[key][:valid_agent_num]], dim=0)
            buffer[key] = torch.cat([x[:valid_sample_num] for x in buffer[key][:valid_agent_num]], dim=0)
            print (out_domain_validation_buffer[key].shape[0], in_domain_validation_buffer[key].shape[0], buffer[key].shape[0])
        else:
            buffer[key] = torch.cat(buffer[key], dim=0)
            print (buffer[key].shape)

    if teacher_mode == 'MT':
        # if the teacher is trained by multi-task RL, reuse its obs rms for the student
        with open(f'expert_data/{source_folder}/obs_rms.pkl', 'rb') as f:
            obs_rms = pickle.load(f)
        if len(cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES) == 6:
            dims = list(range(13)) + [30, 31] + [41, 42]
            new_mean = obs_rms['proprioceptive'].mean.reshape(cfg.MODEL.MAX_LIMBS, -1)
            obs_rms['proprioceptive'].mean = new_mean[:, dims].ravel()
            new_var = obs_rms['proprioceptive'].var.reshape(cfg.MODEL.MAX_LIMBS, -1)
            obs_rms['proprioceptive'].var = new_var[:, dims].ravel()
    elif teacher_mode == 'ST':
        # if the teachers are trained by single-task RL, renormalize the proprioceptive features
        obs_rms = {'proprioceptive': RunningMeanStd(shape=buffer['obs'].shape[-1])}
        obs_rms['proprioceptive'].mean = buffer['obs'].mean(axis=0)
        obs_rms['proprioceptive'].var = buffer['obs'].var(axis=0)
        buffer['obs'] = np.clip(
            (buffer['obs'] - obs_rms['proprioceptive'].mean) / np.sqrt(obs_rms['proprioceptive'].var + 1e-8), 
            -10., 
            10.
        )

    train_data = DistillationDataset(buffer)
    train_dataloader = DataLoader(train_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    if validation:
        in_domain_validation_data = DistillationDataset(in_domain_validation_buffer)
        in_domain_validation_dataloader = DataLoader(in_domain_validation_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        out_domain_validation_data = DistillationDataset(out_domain_validation_buffer)
        out_domain_validation_dataloader = DataLoader(out_domain_validation_data, batch_size=cfg.DISTILL.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    if cfg.DISTILL.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg.DISTILL.BASE_LR, 
            eps=cfg.DISTILL.EPS, 
            weight_decay=cfg.DISTILL.WEIGHT_DECAY
        )
    elif cfg.DISTILL.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=cfg.DISTILL.BASE_LR, 
            eps=cfg.DISTILL.EPS, 
            weight_decay=cfg.DISTILL.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unsupported optimizer type")

    def loss_function(obs_dict, act, act_mean):
        if cfg.DISTILL.IMITATION_TARGET == 'act':
            _, pi, logp, _ = model(obs_dict, act=act.cuda(), compute_val=False, unimal_ids=unimal_ids)
        else:
            _, pi, logp, _ = model(obs_dict, act=act_mean.cuda(), compute_val=False, unimal_ids=unimal_ids)
        if cfg.DISTILL.LOSS_TYPE == 'KL':
            if cfg.DISTILL.KL_TARGET == 'act':
                target = act
            elif cfg.DISTILL.KL_TARGET == 'act_mean':
                target = act_mean
            else:
                raise ValueError("Unsupported loss type")
            if cfg.PPO.TANH == 'action':
                pred = torch.tanh(model.action_mu)
                target = torch.tanh(target)
            else:
                pred = model.action_mu
            if cfg.DISTILL.SAMPLE_WEIGHT:
                threshold = cfg.DISTILL.LARGE_ACT_DECAY
                w = torch.where(target.abs() > threshold, torch.exp(threshold - target.abs()), 1.).cuda()
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = 0.5 * (((pred - target.cuda()).square() * w * (1 - obs_dict['act_padding_mask'])).sum(dim=1) / (1 - obs_dict['act_padding_mask']).sum(dim=1)).mean()
                else:
                    loss = 0.5 * ((pred - target.cuda()).square() * w * (1 - obs_dict['act_padding_mask'])).mean()
            else:
                if cfg.DISTILL.BALANCED_LOSS:
                    loss = 0.5 * (((pred - target.cuda()).square() * (1 - obs_dict['act_padding_mask'])).sum(dim=1) / (1 - obs_dict['act_padding_mask']).sum(dim=1)).mean()
                else:
                    loss = 0.5 * ((pred - target.cuda()).square() * (1 - obs_dict['act_padding_mask'])).mean()
        elif cfg.DISTILL.LOSS_TYPE == 'logp':
            if cfg.DISTILL.BALANCED_LOSS:
                loss = -((model.limb_logp * (1 - obs_dict['act_padding_mask'])).sum(dim=1, keepdim=True) / (1 - obs_dict['act_padding_mask']).sum(dim=1, keepdim=True)).mean()
            else:
                loss = -logp.mean()
        else:
            raise ValueError("Unsupported loss type")
        return loss

    loss_curve, in_domain_valid_curve, out_domain_valid_curve = [], [], []
    for i in range(cfg.DISTILL.EPOCH_NUM):

        if i % cfg.DISTILL.SAVE_FREQ == 0:
            torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{i}.pt')
        elif i <= 50:
            if i % 5 == 0:
                torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{i}.pt')

        batch_losses = []
        for j, (obs, train_act, train_act_mean, hfield, unimal_ids) in enumerate(train_dataloader):

            context = all_context['context'][unimal_ids]
            obs_mask = all_context['obs_mask'][unimal_ids]
            act_mask = all_context['act_mask'][unimal_ids]
            adjacency_matrix = all_context['adjacency_matrix'][unimal_ids]

            train_obs_dict = {
                'proprioceptive': obs.cuda(), 
                'context': context, 
                'obs_padding_mask': obs_mask,  
                'act_padding_mask': act_mask, 
                'adjacency_matrix': adjacency_matrix, 
            }
            if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                train_obs_dict['hfield'] = hfield.cuda()

            loss = loss_function(train_obs_dict, train_act, train_act_mean)
            # if cfg.DISTILL.BASE_WEIGHT_DECAY is not None:
            #     if j % 100 == 0:
            #         print (f'batch {j}: loss = {loss.item()}, L2 reg = {model.mu_net.base_norm_square}')
            #     loss += cfg.DISTILL.BASE_WEIGHT_DECAY * model.mu_net.base_norm_square

            optimizer.zero_grad()
            loss.backward()
            if cfg.DISTILL.GRAD_NORM is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.DISTILL.GRAD_NORM)
            optimizer.step()
            batch_losses.append(loss.item())

        if validation:
            batch_in_domain_valid_loss = []
            for obs, valid_act, valid_act_mean, context, obs_mask, act_mask, hfield, unimal_ids in in_domain_validation_dataloader:
                valid_obs_dict = {
                    'proprioceptive': obs.cuda(), 
                    'context': context.cuda(), 
                    'obs_padding_mask': obs_mask.cuda(), 
                    'act_padding_mask': act_mask.cuda(), 
                }
                if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                    valid_obs_dict['hfield'] = hfield.cuda()
                with torch.no_grad():
                    loss = loss_function(valid_obs_dict, valid_act, valid_act_mean)
                batch_in_domain_valid_loss.append(loss.item())
            in_domain_valid_curve.append(np.mean(batch_in_domain_valid_loss))

            batch_out_domain_valid_loss = []
            for obs, valid_act, valid_act_mean, context, obs_mask, act_mask, hfield, unimal_ids in out_domain_validation_dataloader:
                valid_obs_dict = {
                    'proprioceptive': obs.cuda(), 
                    'context': context.cuda(), 
                    'obs_padding_mask': obs_mask.cuda(), 
                    'act_padding_mask': act_mask.cuda(), 
                }
                if 'hfield' in cfg.ENV.KEYS_TO_KEEP:
                    valid_obs_dict['hfield'] = hfield.cuda()
                with torch.no_grad():
                    loss = loss_function(valid_obs_dict, valid_act, valid_act_mean)
                batch_out_domain_valid_loss.append(loss.item())
            out_domain_valid_curve.append(np.mean(batch_out_domain_valid_loss))

            print (f'epoch {i}, train: {np.mean(batch_losses):.4f}, in domain valid: {np.mean(batch_in_domain_valid_loss):.4f}, out domain valid: {np.mean(batch_out_domain_valid_loss):.4f}')
        else:
            print (f'epoch {i}, average batch loss: {np.mean(batch_losses)}')
        params_norm = torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), 2).item()
        print ('model norm: ', params_norm)
        loss_curve.append(np.mean(batch_losses))
        with open(f'{cfg.OUT_DIR}/loss_curve.pkl', 'wb') as f:
            pickle.dump([loss_curve, in_domain_valid_curve, out_domain_valid_curve], f)

    torch.save([model.mu_net.state_dict(), obs_rms], f'{cfg.OUT_DIR}/checkpoint_{cfg.DISTILL.EPOCH_NUM}.pt')
