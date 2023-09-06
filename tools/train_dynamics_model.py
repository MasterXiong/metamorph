import os
import torch
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import time

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.dynamics import *

from tools.train_ppo import set_cfg_options

torch.manual_seed(0)

MAX_LIMB_NUM = 12
PROPRIOCEPTIVE_OBS_DIM = np.concatenate([list(range(13)), [30, 31], [41, 42]])


class DynamicsDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['obs_act_tuple'])

    def __getitem__(self, index):
        obs_act_tuple = self.data['obs_act_tuple'][index]
        next_obs = self.data['next_obs'][index]
        obs_mask = self.data['obs_mask'][index]
        return obs_act_tuple, next_obs, obs_mask


def collect_trajectory(policy, env, rollout_length=1000):

    obs_buffer = []
    next_obs_buffer = []
    action_buffer = []
    done_buffer = []

    obs = env.reset()
    act_mask = obs["act_padding_mask"].bool()
    obs_mask = obs["obs_padding_mask"].bool()

    for _ in range(rollout_length):
        _, act, _, _, _ = policy.act(obs, return_attention=False, compute_val=False)
        next_obs, reward, done, infos = env.step(act)
        obs_buffer.append(obs['proprioceptive'])
        next_obs_buffer.append(next_obs['proprioceptive'])
        done_buffer.append(torch.Tensor(done))

        act[act_mask] = 0.0
        act = torch.clip(act, -1., 1.)
        action_buffer.append(act)

        obs = next_obs
    
    obs_buffer = torch.stack(obs_buffer)
    next_obs_buffer = torch.stack(next_obs_buffer)
    action_buffer = torch.stack(action_buffer)
    done_buffer = torch.stack(done_buffer)

    return obs_buffer, next_obs_buffer, action_buffer, done_buffer, obs_mask


def collect_data(model_path, agent_path, agent_range):

    agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    cfg.merge_from_file(f'output/{model_path}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'output/{model_path}/Unimal-v0.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.PPO.NUM_ENVS = 32
    set_cfg_options()

    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    policy.ac.eval()

    os.makedirs(f'dynamics_data/{model_path}', exist_ok=True)

    for i in agent_range:
        agent = agents[i]
        print (agent)
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        # set_ret_rms(envs, get_ret_rms(ppo_trainer.envs))
        obs_buffer, next_obs_buffer, action_buffer, done_buffer, obs_mask = collect_trajectory(policy, envs)
        envs.close()

        with open(f'dynamics_data/{model_path}/{agent}.pkl', 'wb') as f:
            pickle.dump([obs_buffer, next_obs_buffer, action_buffer, done_buffer, obs_mask], f)


def preprocess_dynamics_data(data_path):
    with open(data_path, 'rb') as f:
        obs_buffer, next_obs_buffer, action_buffer, done_buffer, obs_mask = pickle.load(f)
    data_size = obs_buffer.shape[0] * obs_buffer.shape[1]
    # obs_buffer: rollout_length * num_env * feat_dim -> full_batch_size * seq_len * limb_feat_dim
    obs = obs_buffer.reshape(data_size, -1).reshape(data_size, MAX_LIMB_NUM, -1)
    # act_buffer: rollout_length * num_env * act_dim -> full_batch_size * seq_len * limb_act_dim
    act = action_buffer.reshape(data_size, -1).reshape(data_size, MAX_LIMB_NUM, -1)
    # concatenate obs and act
    obs_act_tuple = torch.cat([obs, act], -1)
    # reshape next_obs and remove context features
    next_obs = next_obs_buffer.reshape(data_size, -1).reshape(data_size, MAX_LIMB_NUM, -1)
    next_obs = next_obs[:, :, PROPRIOCEPTIVE_OBS_DIM]
    # filter out last step's transition
    idx = ~done_buffer.bool().reshape(data_size)
    obs_act_tuple = obs_act_tuple[idx]
    next_obs = next_obs[idx]
    # reshape obs_mask
    obs_mask = torch.stack([obs_mask[0] for _ in range(obs_act_tuple.shape[0])])

    return obs_act_tuple.detach().cpu(), next_obs.detach().cpu(), obs_mask.detach().cpu()


def load_data(model_path):

    agents = [x[:-4] for x in os.listdir(f'unimals_100/train_mutate_1000/xml')]
    train_agents = agents[:int(len(agents) * 0.8)]
    test_agents = agents[int(len(agents) * 0.8):]

    train_data, in_domain_test_data, out_domain_test_data = defaultdict(list), defaultdict(list), defaultdict(list)
    for agent in train_agents:
        print (agent)
        data_path = f'dynamics_data/{model_path}/{agent}.pkl'
        obs_act_tuple, next_obs, obs_mask = preprocess_dynamics_data(data_path)
        
        test_index = np.random.choice(obs_act_tuple.shape[0], int(obs_act_tuple.shape[0] * 0.2))
        idx = torch.zeros(obs_act_tuple.shape[0])
        idx[test_index] = 1
        idx = idx.bool()
        train_data['obs_act_tuple'].append(obs_act_tuple[~idx])
        train_data['next_obs'].append(next_obs[~idx])
        train_data['obs_mask'].append(obs_mask[~idx])
        in_domain_test_data['obs_act_tuple'].append(obs_act_tuple[idx])
        in_domain_test_data['next_obs'].append(next_obs[idx])
        in_domain_test_data['obs_mask'].append(obs_mask[idx])
    
    for agent in test_agents:
        print (agent)
        data_path = f'dynamics_data/{model_path}/{agent}.pkl'
        obs_act_tuple, next_obs, obs_mask = preprocess_dynamics_data(data_path)
        out_domain_test_data['obs_act_tuple'].append(obs_act_tuple)
        out_domain_test_data['next_obs'].append(next_obs)
        out_domain_test_data['obs_mask'].append(obs_mask)

    for key in ['obs_act_tuple', 'next_obs', 'obs_mask']:
        train_data[key] = torch.cat(train_data[key])
        in_domain_test_data[key] = torch.cat(in_domain_test_data[key])
        out_domain_test_data[key] = torch.cat(out_domain_test_data[key])
    
    with open(f'dynamics_data/{model_path}/all_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'dynamics_data/{model_path}/all_in_domain_test.pkl', 'wb') as f:
        pickle.dump(in_domain_test_data, f)
    with open(f'dynamics_data/{model_path}/all_out_domain_test.pkl', 'wb') as f:
        pickle.dump(out_domain_test_data, f)
    
    return train_data, in_domain_test_data, out_domain_test_data



if __name__ == "__main__":
    
    # generate rollout data
    model_path = 'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout/1409'
    # agent_path = 'unimals_100/train_mutate_1000'
    # collect_data(model_path, agent_path, list(range(750, 1000)))

    # train
    if not os.path.exists(f'dynamics_data/{model_path}/all_train.pkl'):
        train_data, in_domain_test_data, out_domain_test_data = load_data(model_path)
    else:
        with open(f'dynamics_data/{model_path}/all_train.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(f'dynamics_data/{model_path}/all_in_domain_test.pkl', 'rb') as f:
            in_domain_test_data = pickle.load(f)
        with open(f'dynamics_data/{model_path}/all_out_domain_test.pkl', 'rb') as f:
            out_domain_test_data = pickle.load(f)

    trainer = DynamicsTrainer(train_data['obs_act_tuple'].shape[-1], train_data['next_obs'].shape[-1])
    with open(f'dynamics_model/{model_path}/checkpoint_100.pt', 'rb') as f:
        model_state_dict, optimizer_state_dict = pickle.load(f)
    trainer.model.load_state_dict(model_state_dict)
    trainer.optimizer.load_state_dict(optimizer_state_dict)
    torch.manual_seed(10)
    start_epoch = 100
    with open(f'dynamics_model/{model_path}/loss.pkl', 'wb') as f:
        train_loss_curve, in_domain_test_loss_curve, out_domain_test_loss_curve = pickle.load(f)

    train_data = DynamicsDataset(train_data)
    in_domain_test_data = DynamicsDataset(in_domain_test_data)
    out_domain_test_data = DynamicsDataset(out_domain_test_data)

    train_dataloader = DataLoader(train_data, batch_size=cfg.DYNAMICS.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    in_domain_test_dataloader = DataLoader(in_domain_test_data, batch_size=cfg.DYNAMICS.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    out_domain_test_dataloader = DataLoader(out_domain_test_data, batch_size=cfg.DYNAMICS.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)

    # train_loss_curve, in_domain_test_loss_curve, out_domain_test_loss_curve = [], [], []
    os.makedirs(f'dynamics_model/{model_path}', exist_ok=True)
    for i in range(start_epoch, start_epoch + cfg.DYNAMICS.EPOCH_NUM):

        print (f'epoch {i}')

        train_batch_loss = []
        train_start_time = time.time()
        total_train_time = 0.
        for j, (obs_act_tuple, next_obs, obs_mask) in enumerate(train_dataloader):
            batch_train_start = time.time()
            loss = trainer.train_on_batch(obs_act_tuple.cuda(), next_obs.cuda(), obs_mask.cuda())
            batch_train_end = time.time()
            total_train_time += (batch_train_end - batch_train_start)
            train_batch_loss.append(loss)
            if j % 100 == 0:
                print (f'{j} minibatches finished, batch loss: {loss:.6f}')
                total_time_till_now = time.time() - train_start_time
                ratio = total_train_time / total_time_till_now
                time_per_batch = total_time_till_now / (j + 1)
                print (f'{time_per_batch:.2f} seconds per batch, model training time fraction: {ratio:.4f}')
        train_loss_curve.append(np.mean(train_batch_loss))

        in_domain_test_batch_loss = []
        for obs_act_tuple, next_obs, obs_mask in in_domain_test_dataloader:
            obs_act_tuple, next_obs, obs_mask = obs_act_tuple.cuda(), next_obs.cuda(), obs_mask.cuda()
            with torch.no_grad():
                next_obs_pred = trainer.model(obs_act_tuple, obs_mask)
                loss = (((next_obs_pred - next_obs) ** 2) * ~obs_mask[:, :, None]).mean().item()
            in_domain_test_batch_loss.append(loss)
        in_domain_test_loss_curve.append(np.mean(in_domain_test_batch_loss))

        out_domain_test_batch_loss = []
        for obs_act_tuple, next_obs, obs_mask in out_domain_test_dataloader:
            obs_act_tuple, next_obs, obs_mask = obs_act_tuple.cuda(), next_obs.cuda(), obs_mask.cuda()
            with torch.no_grad():
                next_obs_pred = trainer.model(obs_act_tuple, obs_mask)
                loss = (((next_obs_pred - next_obs) ** 2) * ~obs_mask[:, :, None]).mean().item()
            out_domain_test_batch_loss.append(loss)
        out_domain_test_loss_curve.append(np.mean(out_domain_test_batch_loss))

        print (f'train loss: {train_loss_curve[-1]:.6f}, in domain test loss: {in_domain_test_loss_curve[-1]:.6f}, out domain test loss: {out_domain_test_loss_curve[-1]:.6f}')

        if (i + 1) % 10 == 0:
            with open(f'dynamics_model/{model_path}/loss.pkl', 'wb') as f:
                pickle.dump([train_loss_curve, in_domain_test_loss_curve, out_domain_test_loss_curve], f)
            torch.save([trainer.model.state_dict(), trainer.optimizer.state_dict()], f'dynamics_model/{model_path}/checkpoint_{i+1}.pt')

    # with open(f'dynamics_model/{model_path}/loss.pkl', 'wb') as f:
    #     pickle.dump([train_loss_curve, in_domain_test_loss_curve, out_domain_test_loss_curve], f)
    # torch.save([trainer.model.state_dict(), trainer.optimizer.state_dict()], f'dynamics_model/{model_path}/checkpoint_final.pt')
