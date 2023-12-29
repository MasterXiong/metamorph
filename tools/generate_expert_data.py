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


def generate_expert_data(model_path, agent_path, start=0, end=100):

    data_path = 'expert_data/' + '/'.join(model_path.split('/')[1:])
    os.makedirs(data_path, exist_ok=True)

    cfg.merge_from_file(f'{model_path}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'{model_path}/Unimal-v0.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    set_cfg_options()
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    policy.ac.eval()

    def collect_data(agent, num_env=64, timesteps=1000):
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True, num_env=num_env, seed=0)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))

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

        with open(f'{data_path}/{agent}.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        return np.mean(score)

    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')][start:end]
    all_agent_scores = []
    for agent in agents:
        score = collect_data(agent, timesteps=1000)
        print (score)
        all_agent_scores.append(score)
    print ('avg expert score: ', np.mean(all_agent_scores))
    
    with open(f'{data_path}/obs_rms.pkl', 'wb') as f:
        pickle.dump(get_ob_rms(ppo_trainer.envs), f)



if __name__ == '__main__':

    # python tools/generate_expert_data.py --model_path baselines/ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout/1409 --start 0 --end 25
    parser = argparse.ArgumentParser(description="Collect expert data from trained RL agent")
    parser.add_argument("--model_path", help="the path of the expert", required=True, type=str)
    parser.add_argument("--start", help="the start index", type=int, default=0)
    parser.add_argument("--end", help="the end index", type=int, default=100)
    args = parser.parse_args()

    model_path = args.model_path
    agent_path = 'data/train'
    generate_expert_data(model_path, agent_path, start=args.start, end=args.end)
