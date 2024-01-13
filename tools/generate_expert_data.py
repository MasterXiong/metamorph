import argparse
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


def collect_data(agent, ppo_trainer, data_path, num_env=64, timesteps=1000, denormalize=False):
    envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True, num_env=num_env, seed=0)
    obs_rms = get_ob_rms(ppo_trainer.envs)
    set_ob_rms(envs, obs_rms)

    policy = ppo_trainer.agent
    policy.ac.eval()

    obs_buffer = torch.zeros(timesteps, num_env, envs.observation_space['proprioceptive'].shape[0], device='cuda')
    act_buffer = torch.zeros(timesteps, num_env, envs.action_space.shape[0], device='cuda')
    act_mean_buffer = torch.zeros(timesteps, num_env, envs.action_space.shape[0], device='cuda')
    try:
        hfield_buffer = torch.zeros(timesteps, num_env, envs.observation_space['hfield'].shape[0], device='cuda')
    except:
        pass

    obs = envs.reset()
    score = []
    for t in range(timesteps):
        _, act, _ = policy.act(obs, return_attention=False, compute_val=False)
        obs_buffer[t] = obs['proprioceptive'].clone()
        act_buffer[t] = act.detach().clone()
        act_mean_buffer[t] = policy.pi.loc.detach().clone()
        try:
            hfield_buffer[t] = obs['hfield'].clone()
        except:
            pass
        if cfg.PPO.TANH == 'action':
            obs, reward, done, infos = envs.step(torch.tanh(act))
        else:
            obs, reward, done, infos = envs.step(act)
        for info in infos:
            if 'episode' in info:
                score.append(info['episode']['r'])
    envs.close()

    data = {
        'obs': obs_buffer.view(-1, obs_buffer.shape[-1]).cpu(), 
        'act': act_buffer.view(-1, act_buffer.shape[-1]).cpu(), 
        'act_mean': act_mean_buffer.view(-1, act_mean_buffer.shape[-1]).cpu(), 
    }
    try:
        data['hfield'] = hfield_buffer.view(-1, hfield_buffer.shape[-1]).cpu()
    except:
        pass
    obs_clip_ratio = ((data['obs'].abs() - 10).abs() < 0.1).float().mean().item()
    print ('normalization clip ratio: ', obs_clip_ratio)
    if denormalize:
        mean = obs_rms['proprioceptive'].mean
        var = obs_rms['proprioceptive'].var
        epsilon = 1e-8
        data['obs'] = (data['obs'] * np.sqrt(var + epsilon) + mean).float()

    with open(f'{data_path}/{agent}.pkl', 'wb') as f:
        pickle.dump(data, f)
    return np.mean(score), obs_clip_ratio


def generate_expert_data(model_path, agent_path, start=0, end=100, seed='1409', mode='MT'):

    data_path = '/'.join(['expert_data', model_path.split('/')[1], seed])
    os.makedirs(data_path, exist_ok=True)
    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')][start:end]
    all_agent_scores = []

    if mode == 'MT':
        cfg.merge_from_file(f'{model_path}/{seed}/config.yaml')
        cfg.PPO.CHECKPOINT_PATH = f'{model_path}/{seed}/Unimal-v0.pt'
        cfg.ENV.WALKERS = []
        cfg.ENV.WALKER_DIR = agent_path
        set_cfg_options()
        ppo_trainer = PPO()

        for agent in agents:
            score, _ = collect_data(agent, ppo_trainer, data_path, denormalize=False)
            print (agent, score)
            all_agent_scores.append(score)
        print ('avg expert score: ', np.mean(all_agent_scores))
        
        with open(f'{data_path}/obs_rms.pkl', 'wb') as f:
            pickle.dump(get_ob_rms(ppo_trainer.envs), f)
    
    elif mode == 'ST':
        ratio = []
        for agent in agents:
            cfg.merge_from_file(f'{model_path}/{agent}/{seed}/config.yaml')
            cfg.PPO.CHECKPOINT_PATH = f'{model_path}/{agent}/{seed}/Unimal-v0.pt'
            cfg.ENV.WALKERS = []
            cfg.ENV.WALKER_DIR = agent_path
            set_cfg_options()
            ppo_trainer = PPO()
            score, obs_clip_ratio = collect_data(agent, ppo_trainer, data_path, denormalize=True)
            print (agent, score)
            all_agent_scores.append(score)
            ratio.append(obs_clip_ratio)
        print ('avg expert score: ', np.mean(all_agent_scores))
        print ('obs clip ratio')
        print (ratio)



if __name__ == '__main__':

    # MT
    # python tools/generate_expert_data.py --model_path baselines/csr_200M_HN+FA_KL_3_wo_PE+dropout --mode MT --seed 1409 --start 0 --end 25
    # ST
    # python tools/generate_expert_data.py --model_path output/MLP_ST_ft_256*2_KL_5_tanh_action_no_context_in_state --seed 1409 --mode ST --start 0 --end 100
    parser = argparse.ArgumentParser(description="Collect expert data from trained RL agent")
    parser.add_argument("--model_path", help="the path of the expert", required=True, type=str)
    parser.add_argument("--seed", help="seed of the expert", required=True, type=str)
    parser.add_argument("--mode", help="multi-task or single-task expert", required=True, type=str)
    parser.add_argument("--start", help="the start index", type=int, default=0)
    parser.add_argument("--end", help="the end index", type=int, default=100)
    args = parser.parse_args()

    model_path = args.model_path
    agent_path = 'data/train'
    generate_expert_data(model_path, agent_path, start=args.start, end=args.end, seed=args.seed, mode=args.mode)
