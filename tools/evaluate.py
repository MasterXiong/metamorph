# python tools/evaluate.py --policy_path log_origin --policy_name Unimal-v0
import argparse
import os
import torch
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
    '''
    for key in obs:
        print (key, obs[key].size())
        if key != 'proprioceptive':
            print (obs[key][0])
    '''
    for t in range(2000):
        if t % 100 == 0:
            x = obs['proprioceptive'].cpu().numpy()
            x = x.ravel()
            print (f'step {t}: mean: {x.mean()}, var: {x.var()}, clip ratio: {((x == 10.).sum() + (x == -10.).sum()) / len(x)}')
            # print (len(x[x!=0])/len(x))
            plt.figure()
            plt.hist(x[x != 0], bins=100)
            plt.savefig(cfg.OUT_DIR + f'/obs_dist_{agent}_{t}.png')
            plt.close()
        _, act, _ = policy.act(obs)
        obs, reward, done, infos = env.step(act)
        idx = np.where(done)[0]
        #if done or 'episode' in infos:
        #    print (done, infos)
        for i in idx:
            if not_done[i] == 1:
                not_done[i] = 0
                episode_return[i] = infos[i]['episode']['r']
        if not_done.sum() == 0:
            break
    #print (infos)
    avg_reward = episode_return.mean()
    #print (infos[0]['name'], avg_reward)
    return avg_reward


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
def record_video(model_path, test_folder, output_path):
    # cfg.MODEL.FINETUNE.FULL_MODEL = True
    cfg.PPO.CHECKPOINT_PATH = model_path
    #cfg.ENV.WALKER_DIR = './unimals_100/train'
    cfg.ENV.WALKER_DIR = test_folder
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = output_path

    cfg.VECENV.TYPE = "DummyVecEnv"
    cfg.PPO.NUM_ENVS = 1
    cfg.VECENV.IN_SERIES = 1
    
    cfg.merge_from_file('./configs/ft.yaml')
    set_cfg_options()
    #print (cfg.ENV.WALKERS)
    ppo_trainer = PPO()
    cfg.PPO.VIDEO_LENGTH = 1000
    policy = ppo_trainer.agent
    ppo_trainer.save_video(cfg.OUT_DIR)


def evaluate_model(model_path, agent_path):

    test_agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.merge_from_file('./configs/ft.yaml')
    set_cfg_options()
    ppo_trainer = PPO()
    policy = ppo_trainer.agent

    for agent in test_agents:
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        r = evaluate(policy, envs, agent)
        envs.close()
        print (agent, r)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--policy_path", required=True, type=str)
    parser.add_argument("--policy_name", default='Unimal-v0', type=str)
    args = parser.parse_args()
    
    # record videos of zero-shot transfer to test robots
    model_path = os.path.join(args.policy_path, args.policy_name + '.pt')
    evaluate_model(model_path, 'unimals_100/test')
    # for agent in os.listdir('unimals_single_task_test'):
    #     test_folder = os.path.join('unimals_single_task_test', agent)
    #     output_path = f'output/eval/zero_shot_transfer/{agent}'
        # record_video(model_path, test_folder, output_path)

    
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
