import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.model import ActorCritic, Agent

from tools.train_ppo import set_cfg_options

torch.manual_seed(0)
np.random.seed(0)


def generate_expert_data(model_path, agent_path):

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

    def collect_data(agent, num_env=64, timesteps=2000):
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True, num_env=num_env, seed=0)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        set_ret_rms(envs, get_ret_rms(ppo_trainer.envs))

        obs_buffer = torch.zeros(timesteps, num_env, envs.observation_space['proprioceptive'].shape[0], device='cuda')
        act_buffer = torch.zeros(timesteps, num_env, envs.action_space.shape[0], device='cuda')
        context_buffer = torch.zeros(timesteps, num_env, envs.observation_space['context'].shape[0], device='cuda')
        obs_mask_buffer = torch.zeros(timesteps, num_env, envs.observation_space['obs_padding_mask'].shape[0], device='cuda')
        act_mask_buffer = torch.zeros(timesteps, num_env, envs.observation_space['act_padding_mask'].shape[0], device='cuda')

        obs = envs.reset()
        score = []
        for t in range(timesteps):
            _, act, _ = policy.act(obs, return_attention=False, compute_val=False)
            obs_buffer[t] = obs['proprioceptive'].clone()
            act_buffer[t] = act.detach().clone()
            context_buffer[t] = obs['context'].clone()
            obs_mask_buffer[t] = obs['obs_padding_mask'].clone()
            act_mask_buffer[t] = obs['act_padding_mask'].clone()
            obs, reward, done, infos = envs.step(act)
            for info in infos:
                if 'episode' in info:
                    score.append(info['episode']['r'])

        envs.close()

        data = {
            'obs': obs_buffer.view(-1, obs_buffer.shape[-1]).cpu(), 
            'act': act_buffer.view(-1, act_buffer.shape[-1]).cpu(), 
            'context': context_buffer.view(-1, context_buffer.shape[-1]).cpu(), 
            'obs_mask': obs_mask_buffer.view(-1, obs_mask_buffer.shape[-1]).cpu(), 
            'act_mask': act_mask_buffer.view(-1, act_mask_buffer.shape[-1]).cpu(), 
        }

        with open(f'{data_path}/{agent}.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        return np.mean(score)

    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')]
    all_agent_scores = []
    for agent in agents:
        # collect_data(agent, num_env=2, timesteps=100)
        score = collect_data(agent)
        print (score)
        all_agent_scores.append(score)
    print ('avg expert score: ', np.mean(all_agent_scores))
    
    with open(f'{data_path}/obs_rms.pkl', 'wb') as f:
        pickle.dump(get_ob_rms(ppo_trainer.envs), f)


class DistillationDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, index):
        obs = self.data['obs'][index]
        act = self.data['act'][index]
        context = self.data['context'][index]
        obs_mask = self.data['obs_mask'][index]
        act_mask = self.data['act_mask'][index]
        return obs, act, context, obs_mask, act_mask


def distill_policy(source_folder, target_folder, agents):

    cfg.ENV.WALKER_DIR = 'data/train'
    cfg.ENV.WALKERS = agents
    cfg.DISTILL.PER_AGENT_SAMPLE_NUM = 16000
    set_cfg_options()
    envs = make_vec_envs()
    model = ActorCritic(envs.observation_space, envs.action_space)

    # merge the training data
    buffer = {
        'obs': [], 
        'act': [], 
        'context': [], 
        'obs_mask': [], 
        'act_mask': [], 
    }
    for agent in agents:
        data_path = f'expert_data/{source_folder}/{agent}.pkl'
        with open(data_path, 'rb') as f:
            agent_data = pickle.load(f)
        for key in agent_data:
            buffer[key].append(agent_data[key][:cfg.DISTILL.PER_AGENT_SAMPLE_NUM])
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

    output_path = f'distilled_policy/{target_folder}'
    os.makedirs(output_path, exist_ok=True)

    for i in range(cfg.DISTILL.EPOCH_NUM):

        if i % 10 == 0:
            torch.save(model.state_dict(), f'{output_path}/checkpoint_{i}.pt')

        batch_losses = []
        for obs, act, context, obs_mask, act_mask in train_dataloader:

            obs_dict = {
                'proprioceptive': obs, 
                'context': context, 
                'obs_padding_mask': obs_mask, 
                'act_padding_mask': act_mask, 
            }

            _, pi, logp, _ = model(obs_dict, act=act, compute_val=False)
            loss = -logp.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss)

        print (f'epoch {i}, average batch loss: {np.mean(batch_losses)}')

    torch.save(model.state_dict(), f'{output_path}/checkpoint_final.pt')
