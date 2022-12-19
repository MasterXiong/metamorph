# python tools/evaluate.py --policy_path log_origin --policy_name Unimal-v0
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.algos.ppo.model import Agent

import time


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    num_joints, num_limbs = [], []

    metadata_paths = []
    for agent in cfg.ENV.WALKERS:
        metadata_paths.append(os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        ))

    for metadata_path in metadata_paths:
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1
    cfg.MODEL.MAX_JOINTS = 16
    cfg.MODEL.MAX_LIMBS = 12
    print (cfg.MODEL.MAX_JOINTS, cfg.MODEL.MAX_LIMBS)


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml"))
    ]


def evaluate(policy, env, agent):
    episode_return = np.zeros(cfg.PPO.NUM_ENVS)
    not_done = np.ones(cfg.PPO.NUM_ENVS)
    obs = env.reset()
    # print ('state')
    # print (obs['proprioceptive'].size())
    # print (obs['proprioceptive'][:4, :10])
    # print ('context')
    # print (obs['context'][:4, :10])
    context = obs['context'][0].cpu().numpy()
    ood_ratio = (np.abs(context) > 1.).mean()
    '''
    for key in obs:
        print (key, obs[key].size())
        if key != 'proprioceptive':
            print (obs[key][0])
    '''
    # attention_curve = [[] for _ in range(5)]
    for t in range(2000):
        # if t % 100 == 0:
        #     x = obs['proprioceptive'].cpu().numpy()
        #     x = x.ravel()
        #     print (f'step {t}: mean: {x.mean()}, var: {x.var()}, clip ratio: {((x == 10.).sum() + (x == -10.).sum()) / len(x)}')
        #     # print (len(x[x!=0])/len(x))
        #     plt.figure()
        #     plt.hist(x[x != 0], bins=100)
        #     plt.savefig(cfg.OUT_DIR + f'/obs_dist_{agent}_{t}.png')
        #     plt.close()
        _, act, _, _, _ = policy.act(obs, return_attention=False)
        obs, reward, done, infos = env.step(act)
        # for i in range(5):
            # attention_curve[i].append(policy.v_attention_maps[i].cpu().numpy())
        # print (policy.v_attention_maps[0])
        # for i in range(obs['context'].shape[0]):
        #     if (obs['context'][i] == context).sum() != obs['context'].shape[1]:
        #         print ('agent changed!')
        idx = np.where(done)[0]
        #if done or 'episode' in infos:
        #    print (done, infos)
        for i in idx:
            if not_done[i] == 1:
                not_done[i] = 0
                episode_return[i] = infos[i]['episode']['r']
                # if 'terminate_on_fall' in infos[i]:
                #     print (f'trial {i} early stops due to falling')
        if not_done.sum() == 0:
            break
    #print (infos[0]['name'], avg_reward)

    # plt.figure()
    # for i in range(12):
    #     att_score = np.stack([x[:, 0, i] for x in attention_curve[-1]])
    #     # plt.plot(att_score.mean(axis=1), label=i)
    #     plt.plot(att_score[:, 0], label=i)
    # plt.legend()
    # plt.savefig('figures/att_score.png')
    # plt.close()

    return episode_return, ood_ratio


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
    agents = agents[:10]

    cfg.OUT_DIR = './eval'
    cfg.merge_from_file('./configs/ft.yaml')

    origin_return = {}
    for agent in agents:
        with open(os.path.join(folder, agent, '1409', 'Unimal-v0_results.json'), 'r') as f:
            results = json.load(f)
            origin_return[agent] = results[agent]['reward']['reward'][-1]

    for source in agents:
        print (source)
        cfg.merge_from_file(f'{folder}/{source}/1409/config.yaml')
        cfg.ENV.WALKER_DIR = 'unimals_100/train'
        cfg.ENV.WALKERS = []
        cfg.OUT_DIR = './eval'
        set_cfg_options()
        policy, ob_rms = torch.load(os.path.join(folder, source, '1409', 'Unimal-v0.pt'))
        policy.v_net.model_args.HYPERNET = False
        policy.mu_net.model_args.HYPERNET = False
        policy = Agent(policy)
        for target in agents:
            envs = make_vec_envs(xml_file=target, training=False, norm_rew=False, render_policy=True)
            set_ob_rms(envs, ob_rms)
            r = evaluate(policy, envs, target)
            envs.close()
            print (f'{source} to {target}: transfer: {r}, origin: {origin_return[target]}')


# record a video of running a trained model on a specific task
def record_video(model_path, config_path, test_folder, output_path):

    cfg.merge_from_file('./configs/ft.yaml')
    cfg.merge_from_file(config_path)

    # cfg.MODEL.FINETUNE.FULL_MODEL = True
    cfg.PPO.CHECKPOINT_PATH = model_path
    #cfg.ENV.WALKER_DIR = './unimals_100/train'
    cfg.ENV.WALKER_DIR = test_folder
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = output_path

    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.IN_SERIES = 1
    
    set_cfg_options()
    #print (cfg.ENV.WALKERS)
    ppo_trainer = PPO()
    cfg.PPO.VIDEO_LENGTH = 1000
    policy = ppo_trainer.agent
    ppo_trainer.save_video(cfg.OUT_DIR)


def evaluate_model(model_path, agent_path, policy_folder, suffix=None, terminate_on_fall=True, deterministic=False):

    test_agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    cfg.merge_from_file('./configs/ft.yaml')
    cfg.merge_from_file(f'{policy_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    # do not include terminating on falling during evaluation
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    cfg.DETERMINISTIC = deterministic
    cfg.PPO.NUM_ENVS = 64
    set_cfg_options()
    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    # change to eval mode as we have dropout in the model
    policy.ac.eval()

    eval_result = {}
    ood_list = np.zeros(len(test_agents))
    avg_score = []
    if len(policy_folder.split('/')) == 3:
        output_name = policy_folder.split('/')[1] + '_' + policy_folder.split('/')[2]
    else:
        output_name = policy_folder.split('/')[1]
    if suffix is not None:
        output_name = f'{output_name}_{suffix}'
    for i, agent in enumerate(test_agents):
        # start = time.time()
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        episode_return, ood_ratio = evaluate(policy, envs, agent)
        envs.close()
        print (agent, f'{episode_return.mean():.2f} +- {episode_return.std():.2f}')
        eval_result[agent] = episode_return
        ood_list[i] = ood_ratio
        avg_score.append(np.array(episode_return).mean())
        with open(f'eval/{output_name}.pkl', 'wb') as f:
            pickle.dump(eval_result, f)
        # plt.figure()
        # plt.scatter(eval_result[:i+1], ood_list[:i+1])
        # plt.savefig('test_ood_context.png')
        # plt.close()
        # end = time.time()
        # print (f'spend {end - start:.2f} seconds on {agent}')
    print ('avg score across all test agents: ', np.array(avg_score).mean())


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

    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--policy_path", required=True, type=str)
    parser.add_argument("--policy_name", default='Unimal-v0', type=str)
    parser.add_argument("--terminate_on_fall", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    # parser.add_argument("--suffix", default=None, type=str)
    args = parser.parse_args()
    
    # example command: python tools/evaluate.py --policy_path output/log_fix_attention_wo_PE/1409 --terminate_on_fall --deterministic
    model_path = os.path.join(args.policy_path, args.policy_name + '.pt')
    suffix = []
    if args.terminate_on_fall:
        suffix.append('terminate_on_fall')
    if args.deterministic:
        suffix.append('deterministic')
    if len(suffix) == 0:
        suffix = None
    else:
        suffix = '_'.join(suffix)
    evaluate_model(model_path, 'unimals_100/test', args.policy_path, suffix=suffix, terminate_on_fall=args.terminate_on_fall, deterministic=args.deterministic)

    # record videos of zero-shot transfer to test robots
    # folder = 'log_baseline_wo_PE'
    # model_path = f'output/{folder}/1409/Unimal-v0.pt'
    # config_path = f'output/{folder}/1409/config.yaml'
    # for agent in os.listdir('unimals_single_task'):
    #     test_folder = os.path.join('unimals_single_task', agent)
    #     output_path = f'output/video/{folder}/{agent}'
    #     record_video(model_path, config_path, test_folder, output_path)

    # folder = 'output/log_single_task'
    # for agent in os.listdir(folder):
    #     print (agent)
    #     model_path = os.path.join(folder, agent, '1409', 'Unimal-v0.pt')
    #     config_path = os.path.join(folder, agent, '1409', 'config.yaml')
    #     test_folder = os.path.join('unimals_single_task', agent)
    #     output_path = f'output/video/{agent}'
    #     record_video(model_path, config_path, test_folder, output_path)
    
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
    # transfer_between_ST('output/log_single_task_wo_pe')
