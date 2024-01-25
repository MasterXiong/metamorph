# python tools/evaluate.py --policy_path log_origin --policy_name Unimal-v0
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import scipy

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.algos.ppo.model import Agent, MLPModel, HNMLP

from tools.train_ppo import set_cfg_options

import time

torch.manual_seed(0)


def compute_GAE(episode_value, episode_reward, timeout=False):
    gamma, gae_lambda = cfg.PPO.GAMMA, cfg.PPO.GAE_LAMBDA
    episode_gae = np.zeros_like(episode_value)
    if timeout:
        episode_gae[-1] = 0.
    else:
        episode_gae[-1] = episode_reward[-1] - episode_value[-1]
    for step in reversed(range(len(episode_value) - 1)):
        delta = episode_reward[step] + gamma * episode_value[step + 1] - episode_value[step]
        episode_gae[step] = delta + gamma * gae_lambda * episode_gae[step + 1]
    return episode_gae


def plot_weights_histogram(params, name, agent, mask):
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        data = params[i].cpu().numpy().ravel()
        if len(data) > 1000:
            bins = 100
        else:
            bins = 20
        if mask[i] == 0:
            color = 'blue'
        else:
            color = 'orange'
        plt.hist(data, bins=bins, color=color)
        plt.title(f'std: {data.std():.4f}')
    plt.tight_layout()
    plt.savefig(f'figures/analyze_HNMLP/{name}_{agent}.png')
    plt.close()


def evaluate(policy, env, agent, compute_gae=False):

    episode_return = np.zeros(cfg.PPO.NUM_ENVS)
    not_done = np.ones(cfg.PPO.NUM_ENVS)
    episode_values, episode_rewards = [], []
    episode_len = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)
    timeout = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)

    obs = env.reset()
    ood_ratio = (obs['proprioceptive'].abs() == 10.).float().mean().item()
    # if cfg.MODEL.NORMALIZE_CONTEXT:
    #     with open('context_norm.pkl', 'rb') as f:
    #         context_min, context_max = pickle.load(f)
    #     context_min = torch.Tensor(context_min).float().unsqueeze(0).cuda()
    #     context_max = torch.Tensor(context_max).float().unsqueeze(0).cuda()
    #     context_range = context_max - context_min
    #     context_range[context_range == 0] = 1e-8
    #     obs_context = obs['context']
    #     batch_size = obs_context.shape[0]
    #     obs_context = obs_context.view(batch_size * 12, -1)
    #     obs_context = (obs_context - context_min) / context_range
    #     obs_context = obs_context.view(batch_size, -1)
    #     obs['context'] = obs_context

    if type(policy.ac.mu_net) == MLPModel or type(policy.ac.mu_net) == HNMLP:
        morphology_info = {}
        morphology_info['adjacency_matrix'] = obs['adjacency_matrix']
        with torch.no_grad():
            policy.ac.mu_net.generate_params(obs['context'], obs['obs_padding_mask'].bool(), morphology_info=morphology_info)

    unimal_ids = env.get_unimal_idx()
    for t in range(2000):
        if compute_gae:
            val, act, _ = policy.act(obs, return_attention=False, compute_val=True, unimal_ids=unimal_ids)
            episode_values.append(val)
        else:
            _, act, _ = policy.act(obs, return_attention=False, compute_val=False, unimal_ids=unimal_ids)
        
        # if t == 0:
        #     # mask = obs['obs_padding_mask'][0]
        #     # plot_weights_histogram(policy.ac.mu_net.context_embedding_input[0], 'input_context', agent, mask)
        #     # plot_weights_histogram(policy.ac.mu_net.input_weight[0], 'input_weight', agent, mask)
        #     # plot_weights_histogram(policy.ac.mu_net.context_embedding_output[0], 'output_context', agent, mask)
        #     # plot_weights_histogram(policy.ac.mu_net.output_weight[0], 'output_weight', agent, mask)
        #     try:
        #         with open(f'weights_limb_id_in_context/{agent}.pkl', 'wb') as f:
        #             data = {
        #                 'input_context': policy.ac.mu_net.context_embedding_input[0], 
        #                 'input_weight': policy.ac.mu_net.input_weight[0], 
        #                 'output_context': policy.ac.mu_net.context_embedding_output[0], 
        #                 'output_weight': policy.ac.mu_net.output_weight[0], 
        #             }
        #             pickle.dump(data, f)
        #     except:
        #         with open('weights_TF_context_encoder/MLP.pkl', 'wb') as f:
        #             data = {
        #                 'input_weight': policy.ac.mu_net.input_layer.weight.data, 
        #                 'output_weight': policy.ac.mu_net.output_layer.weight.data, 
        #             }
        #             pickle.dump(data, f)

        if cfg.PPO.TANH == 'action':
            obs, reward, done, infos = env.step(torch.tanh(act))
        else:
            obs, reward, done, infos = env.step(act)
        if compute_gae:
            episode_rewards.append(reward)
        ood_ratio += (obs['proprioceptive'].abs() == 10.).float().mean().item()

        idx = np.where(done)[0]
        for i in idx:
            if not_done[i] == 1:
                not_done[i] = 0
                episode_return[i] = infos[i]['episode']['r']
                episode_len[i] = t + 1
                timeout[i] = 'timeout' in infos[i]
        if not_done.sum() == 0:
            break    

    ood_ratio /= (t + 1)

    episode_gae = []
    if compute_gae:
        episode_values = torch.stack(episode_values).cpu().numpy()
        episode_rewards = torch.stack(episode_rewards).cpu().numpy()
        for i in range(cfg.PPO.NUM_ENVS):
            episode_value = episode_values[:episode_len[i], i]
            episode_reward = episode_rewards[:episode_len[i], i]
            episode_gae.append(compute_GAE(episode_value, episode_reward, timeout=timeout[i]))

    return episode_return, ood_ratio, episode_gae


# evaluate test set performance at different training checkpoints
def evaluate_checkpoint(model_path, test_agent_path, test_interval=10):

    models = [x for x in os.listdir(model_path) if x.startswith('checkpoint')]
    test_agents = [x.split('.')[0] for x in os.listdir(f'{test_agent_path}/xml')][::test_interval]
    if os.path.exists('test_curve.pkl'):
        with open('test_curve.pkl', 'rb') as f:
            scores = pickle.load(f)
    else:
        scores = []
    scores = []

    cfg.merge_from_file('./configs/ft.yaml')
    cfg.merge_from_file(model_path + '/config.yaml')
    cfg.MODEL.OBS_TYPES = ["proprioceptive", "edges", "obs_padding_mask", "act_padding_mask", "context"]

    for idx in range(10, 1200, 50):
        model = f'checkpoint_{idx}.pt'
        print (model)
        cfg.ENV.WALKERS = []
        cfg.PPO.CHECKPOINT_PATH = os.path.join(model_path, model)
        # cfg.MODEL.FINETUNE.FULL_MODEL = True
        cfg.ENV.WALKER_DIR = test_agent_path
        cfg.OUT_DIR = './eval'
        #cfg.PPO.NUM_ENVS = 1
        #cfg.VECENV.IN_SERIES = 1
        set_cfg_options()
        model_p, ob_rms = torch.load(cfg.PPO.CHECKPOINT_PATH)
        model_p.v_net.model_args = cfg.MODEL.TRANSFORMER
        model_p.mu_net.model_args = cfg.MODEL.TRANSFORMER
        policy = Agent(model_p)

        rewards = np.zeros(len(test_agents))
        for i, agent in enumerate(test_agents):
            envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
            set_ob_rms(envs, ob_rms)
            r = evaluate(policy, envs)
            print (agent, r)
            rewards[i] = r
            envs.close()
        
        print (model, rewards.mean())
        scores.append([model, rewards])
    
    with open(model_path + '/test_curve.pkl', 'wb') as f:
        pickle.dump(scores, f)


# evaluate zero-shot transfer between different robots trained on single tasks
def transfer_between_ST(folder):

    agents = []
    for agent in os.listdir(folder):
        if 'Unimal-v0_results.json' in os.listdir(os.path.join(folder, agent, '1409')):
            agents.append(agent)
    agents = agents[:5]

    origin_return = {}
    for agent in agents:
        with open(os.path.join(folder, agent, '1409', 'Unimal-v0_results.json'), 'r') as f:
            results = json.load(f)
            origin_return[agent] = results[agent]['reward']['reward'][-1]

    for source in agents:
        print (source)
        cfg.merge_from_file(f'{folder}/{source}/1409/config.yaml')
        cfg.PPO.CHECKPOINT_PATH = os.path.join(folder, source, '1409', 'Unimal-v0.pt')
        cfg.ENV.WALKER_DIR = 'unimals_100/train'
        cfg.ENV.WALKERS = []
        cfg.OUT_DIR = './eval'
        set_cfg_options()
        ppo_trainer = PPO()
        cfg.PPO.NUM_ENVS = 32
        policy = ppo_trainer.agent
        # change to eval mode as we have dropout in the model
        policy.ac.eval()
        # policy.v_net.model_args.HYPERNET = False
        # policy.mu_net.model_args.HYPERNET = False
        for target in agents:
            envs = make_vec_envs(xml_file=target, training=False, norm_rew=False, render_policy=True)
            set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
            r, _, _, _ = evaluate(policy, envs, target)
            envs.close()
            print (f'{source} to {target}: transfer: {np.array(r).mean()}, origin: {origin_return[target]}')


def evaluate_model(model_path, agent_path, policy_folder, suffix=None, terminate_on_fall=True, deterministic=False, compute_gae=False):

    test_agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    print (policy_folder)
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    # do not include terminating on falling during evaluation
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    set_cfg_options()
    cfg.PPO.NUM_ENVS = 16
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    # change to eval mode as we have dropout in the model
    policy.ac.eval()

    ood_list = np.zeros(len(test_agents))
    avg_score = []
    if len(policy_folder.split('/')) == 3:
        folder_name = policy_folder.split('/')[1]
    else:
        folder_name = policy_folder.split('/')[0]
    output_name = folder_name + '/' + suffix
    os.makedirs(f'eval/{folder_name}', exist_ok=True)
    print (output_name)

    if os.path.exists(f'eval/{output_name}.pkl'):
        with open(f'eval/{output_name}.pkl', 'rb') as f:
            eval_result = pickle.load(f)
    else:
        eval_result = {}

    obs_rms = get_ob_rms(ppo_trainer.envs)
    try:
        obs_rms['proprioceptive'].mean = obs_rms['proprioceptive'].mean.numpy()
        obs_rms['proprioceptive'].var = obs_rms['proprioceptive'].var.numpy()
    except:
        pass

    all_obs = dict()
    for i, agent in enumerate(test_agents):
        # with open(f'{agent_path}/metadata/{agent}.json', 'r') as f:
        #     limb_num = json.load(f)["num_limbs"]
        if agent in eval_result and len(eval_result[agent]) == 3:
            episode_return, ood_ratio, _ = eval_result[agent]
            avg_score.append(np.array(episode_return).mean())
        else:
            envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True, max_episode_num=1)
            set_ob_rms(envs, obs_rms)
            set_ret_rms(envs, get_ret_rms(ppo_trainer.envs))
            episode_return, ood_ratio, episode_gae = evaluate(policy, envs, agent, compute_gae=compute_gae)
            envs.close()
            # print ([np.maximum(x, 0.).mean() for x in episode_gae])
            eval_result[agent] = [episode_return, ood_ratio, episode_gae]
            ood_list[i] = ood_ratio
            avg_score.append(np.array(episode_return).mean())
            with open(f'eval/{output_name}.pkl', 'wb') as f:
                pickle.dump(eval_result, f)
        print (agent, f'{episode_return.mean():.2f} +- {episode_return.std():.2f}', f'OOD ratio: {ood_ratio}')

    print ('avg score across all test agents: ', np.array(avg_score).mean())
    return np.array(avg_score).mean()


def evaluate_model_v2(model_path, agent_path, policy_folder, suffix=None, deterministic=True):

    print (policy_folder)
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    # do not include terminating on falling during evaluation
    cfg.DETERMINISTIC = deterministic
    set_cfg_options()
    test_agents = cfg.ENV.WALKERS

    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    # change to eval mode as we have dropout in the model
    policy.ac.eval()

    folder_name = policy_folder.split('/')[-2]
    output_name = folder_name + '/' + suffix
    os.makedirs(f'eval/{folder_name}', exist_ok=True)
    print (output_name)

    obs_rms = get_ob_rms(ppo_trainer.envs)
    try:
        obs_rms['proprioceptive'].mean = obs_rms['proprioceptive'].mean.numpy()
        obs_rms['proprioceptive'].var = obs_rms['proprioceptive'].var.numpy()
    except:
        pass

    cfg.ENV.FIX_ENV = True
    cfg.VECENV.IN_SERIES = 6
    envs = make_vec_envs(training=False)
    set_ob_rms(envs, obs_rms)
    obs = envs.reset()

    if type(policy.ac.mu_net) == MLPModel or type(policy.ac.mu_net) == HNMLP:
        policy.ac.mu_net.generate_params(obs['context'], obs['obs_padding_mask'].bool())

    episode_per_agent = 16
    episode_return = [[] for _ in test_agents]
    episode_num = np.zeros(len(test_agents))

    for t in range(1000 * episode_per_agent):
        # unimal_ids = env.get_unimal_idx()
        if t % 1000 == 0:
            print (f'{t} steps finished')
        _, act, _ = policy.act(obs, return_attention=False, compute_val=False, unimal_ids=None)
        obs, reward, done, infos = envs.step(act)

        idx = np.where(done)[0]
        episode_num[idx] += 1
        for i in idx:
            episode_return[i].append(infos[i]['episode']['r'])
        if (episode_num < episode_per_agent).sum() == 0:
            break
    envs.close()

    eval_result = {}
    for i, agent in enumerate(test_agents):
        eval_result[agent] = [np.array(episode_return[i][:episode_per_agent]), None, None]
    with open(f'eval/{output_name}.pkl', 'wb') as f:
        pickle.dump(eval_result, f)

    print ('avg score across all test agents: ', np.mean(np.stack(episode_return)))
    return np.mean(np.stack(episode_return))


# def get_context(agent_path):

#     cfg.merge_from_file('./configs/ft.yaml')
#     cfg.ENV.WALKERS = []
#     cfg.ENV.WALKER_DIR = agent_path
#     cfg.OUT_DIR = './eval'
#     set_cfg_options()

#     all_context = []
#     agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]
#     for i, agent in enumerate(agents):
#         envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
#         obs = envs.reset()
#         context = obs['context'][0].cpu().numpy()
#         all_context.append(context)
#         envs.close()
#     return np.stack(all_context)


# def classify_train_test_context():
#     # train_context = get_context('unimals_100/train')
#     # test_context = get_context('unimals_100/test')
#     # data = np.concatenate([train_context, test_context], axis=0)
#     # label = np.concatenate([np.ones(train_context.shape[0]), np.zeros(test_context.shape[0])])
#     # print (data.shape, label.shape)
#     # # print (data)
#     # with open('train_test_context.pkl', 'wb') as f:
#     #     pickle.dump([data, label], f)

#     with open('train_test_context.pkl', 'rb') as f:
#         data, label = pickle.load(f)
    
#     m = nn.Sequential(
#         nn.Linear(data.shape[1], 128), 
#         nn.ReLU(), 
#         nn.Linear(128, 1), 
#         nn.Sigmoid(), 
#     )
#     loss_function = nn.BCELoss()
#     optimizer = torch.optim.Adam(m.parameters())

#     for _ in range(100):
#         pred = m(torch.Tensor(data))
#         loss = loss_function(pred, torch.Tensor(label.reshape(-1, 1)))
#         print (loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     with open('eval/log_HN_fix_normalization_PE_in_base_1409.pkl', 'rb') as f:
#         test_score = pickle.load(f)
    
#     plt.figure()
#     plt.scatter(test_score, pred.detach().numpy().ravel()[-len(test_score):])
#     plt.savefig('figures/test.png')
#     plt.close()



if __name__ == '__main__':

    # classify_train_test_context()

    # example command: python tools/evaluate.py --policy_path output/ft_train_expand_uniform_fix_lr_KL_5_wo_PE+dropout --test_folder unimals_100/test --seed 1409
    # example command: python tools/evaluate.py --policy_path exploration_HN+FA_KL_3_wo_PE+dropout --test_folder kinematics --seed 1411
    # for modular: python tools/evaluate.py --policy_path output/modular_humanoid_train_TF --policy_name Modular-v0 --test_folder modular/humanoid_test

    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--policy_path", default=None, type=str)
    parser.add_argument("--policy_name", default='Unimal-v0', type=str)
    parser.add_argument("--terminate_on_fall", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    # which folder used for evaluation
    parser.add_argument("--test_folder", default='unimals_100/test', type=str)
    parser.add_argument("--task", default=None, type=str)
    # parser.add_argument("--suffix", default=None, type=str)
    args = parser.parse_args()

    folders = {}
    folders['FT'] = [
        'ft_baseline_KL_5_wo_PE+dropout', 
        'ft_HN+FA_KL_5_wo_PE+dropout', 
        'ft_baseline', 
        'ft_FA_KL_5_wo_PE+dropout', 
        'ft_HN_KL_5_wo_PE+dropout', 
    ]
    folders['VT'] = [
        'csr_200M_baseline_KL_3_wo_PE+dropout', 
        'csr_200M_HN+FA_KL_3_wo_PE+dropout', 
        'csr_200M_baseline', 
        'csr_200M_FA_KL_3_wo_PE+dropout', 
        'csr_200M_HN_KL_3_wo_PE+dropout', 
    ]
    folders['Obstacles'] = [
        'obstacle_200M_baseline_KL_3_wo_PE+dropout', 
        'obstacle_200M_HN+FA_KL_3_wo_PE+dropout', 
        'obstacle_200M_baseline', 
        'obstacle_200M_FA_KL_3_wo_PE+dropout', 
        'obstacle_200M_HN_KL_3_wo_PE+dropout', 
    ]
    folders['Incline'] = [
        'incline_baseline_KL_3_wo_PE+dropout', 
        'incline_HN+FA_KL_5_wo_PE+dropout', 
        'incline_baseline', 
        'incline_FA_KL_5_wo_PE+dropout', 
        'incline_HN_KL_5_wo_PE+dropout', 
    ]
    folders['Exploration'] = [
        'exploration_baseline_KL_3_wo_PE+dropout', 
        'exploration_HN+FA_KL_3_wo_PE+dropout', 
        'exploration_baseline', 
        'exploration_FA_KL_3_wo_PE+dropout', 
        'exploration_HN_KL_3_wo_PE+dropout', 
    ]

    if args.policy_path is None:
        eval_folders = folders[args.task]
    else:
        eval_folders = [args.policy_path]

    # example command: python tools/evaluate.py --policy_path best_models/ft_HN+FA_KL_5_wo_PE+dropout --test_folder dynamics
    # example command: python tools/evaluate.py --seed 1409 --policy_path output/log_fix_attention_wo_PE --terminate_on_fall --deterministic
    suffix = []
    if args.terminate_on_fall:
        suffix.append('terminate_on_fall')
    if args.deterministic:
        suffix.append('deterministic')
    if '/' in args.test_folder:
        suffix.append(args.test_folder.split('/')[1])
    else:
        suffix.append(args.test_folder)
    if 'checkpoint' in args.policy_name:
        iteration = args.policy_name.split('_')[1]
        suffix.append(f'cp_{iteration}')
    if len(suffix) == 0:
        suffix = None
    else:
        suffix = '_'.join(suffix)
    print (suffix)

    if args.seed is not None:
        seeds = [str(args.seed)]
    else:
        seeds = ['1409', '1410', '1411']

    for folder in eval_folders:
        # policy_path = f'best_models/{folder}'
        policy_path = folder
        scores = []
        for seed in seeds:
            model_path = os.path.join(policy_path, seed, args.policy_name + '.pt')
            score = evaluate_model(model_path, args.test_folder, os.path.join(policy_path, seed), suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
            scores.append(score)
        scores = np.stack(scores)
        print ('avg score across seeds: ')
        test_agents = [x.split('.')[0] for x in os.listdir(f'{args.test_folder}/xml')]
        for i, agent in enumerate(test_agents):
            print (f'{agent}: {scores[:, i].mean()} +- {scores[:, i].std()}')
        scores = scores.mean(axis=1)
        print (f'overall: {scores.mean()} +- {scores.std()}')


    # if args.seed is not None:
    #     seeds = [str(args.seed)]
    # else:
    #     seeds = os.listdir(args.policy_path)
    # for seed in seeds:
    #     # for idx in range(100, 1300, 100):
    #     #     model_path = os.path.join(args.policy_path, seed, f'checkpoint_{idx}.pt')
    #     #     suffix = str(idx)
    #     #     evaluate_model(model_path, 'unimals_100/test', os.path.join(args.policy_path, seed), suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
    #     model_path = os.path.join(args.policy_path, seed, args.policy_name + '.pt')
    #     evaluate_model(model_path, f'unimals_100/{args.test_folder}', os.path.join(args.policy_path, seed), suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
        # evaluate_model(model_path, 'unimals_100/train', os.path.join(args.policy_path, seed), suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)
    
    '''
    test_agents = [x.split('.')[0] for x in os.listdir('unimals_100/train/xml')]
    count = 0
    model_p, ob_rms = torch.load(cfg.PPO.CHECKPOINT_PATH)
    policy = Agent(model_p)
    for x in test_agents[50:55]:
        envs = make_vec_envs(xml_file=None, training=False, norm_rew=False, render_policy=False)
        #envs = make_vec_envs_zs()
        set_ob_rms(envs, ob_rms)
        r = evaluate(policy, envs)
        print (x)
        print ('evaluation', r)
        with open('log_origin/Unimal-v0_results.json', 'r') as f:
            result = json.load(f)
        print ('train final', result[x]['reward']['reward'][-5:])
        envs.close()
        count += 1
        if count == 1:
            break
    '''
    # evaluate_checkpoint(args.policy_path, 'unimals_100/test', test_interval=10)
    # transfer_between_ST('output/TF_ST_ft_KL_5_wo_PE+dropout')
