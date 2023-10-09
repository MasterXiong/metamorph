import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO

from tools.train_ppo import set_cfg_options
from tools.evaluate import evaluate_model

import torch
torch.manual_seed(1000)


# record a video of running a trained model on a specific task
def record_video(model_path, config_path, test_folder, output_path):

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

    # cfg.TERMINATE_ON_FALL = False
    
    set_cfg_options()
    ppo_trainer = PPO()
    cfg.PPO.VIDEO_LENGTH = 1000
    cfg.PPO.NUM_ENVS = 1

    # ST_scores = []
    # ST_folder = 'output/ST_MLP_train_mutate_constant_lr'
    # for agent in cfg.ENV.WALKERS:
    #     with open(f'{ST_folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         log = json.load(f)
    #     ST_scores.append(np.array(log[agent]['reward']['reward'][-10:]).mean())
    # order = np.argsort(ST_scores)
    # test_agent_id = np.concatenate([order[:20], order[50:350:10], order[-20:]])
    # cfg.ENV.WALKERS = [cfg.ENV.WALKERS[i] for i in test_agent_id]

    # record_agents = []
    # for robot in cfg.ENV.WALKERS:
    #     # finished = False
    #     # for video in os.listdir(cfg.OUT_DIR):
    #     #     if robot in video:
    #     #         finished = True
    #     #         break
    #     # if finished:
    #     #     continue
    #     with open(f'output/ST_MLP_random_1000_constant_lr/{robot}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     score = np.mean(results[robot]['reward']['reward'][-10:])
    #     if score < 1000:
    #         continue
    #     # returns = ppo_trainer.save_video(cfg.OUT_DIR, xml=robot)
    #     # print (robot, np.array(returns).mean())
    #     record_agents.append(robot)
    
    record_agents = cfg.ENV.WALKERS
    for agent in record_agents:
        
        # print (robot, agent)
        # cfg.merge_from_file(f'output/ST_MLP_random_1000_constant_lr/{robot}/1409/config.yaml')
        # cfg.PPO.CHECKPOINT_PATH = f'output/ST_MLP_random_1000_constant_lr/{robot}/1409/Unimal-v0.pt'  
        # cfg.ENV.WALKER_DIR = test_folder
        # cfg.ENV.WALKERS = []
        # cfg.OUT_DIR = output_path
        # cfg.VECENV.TYPE = "DummyVecEnv"
        # cfg.PPO.NUM_ENVS = 1
        # cfg.VECENV.IN_SERIES = 1      
        # set_cfg_options()
        # ppo_trainer = PPO()
        returns = ppo_trainer.save_video(cfg.OUT_DIR, xml=agent)


def analyze():

    cfg.merge_from_file('output/ft_train_mutate_UED_KL_5_wo_PE+dropout/1409/config.yaml')
    # cfg.ENV.WALKER_DIR = test_folder
    cfg.ENV.WALKERS = []
    set_cfg_options()

    ST_scores = []
    ST_folder = 'output/ST_MLP_train_mutate_constant_lr'
    for agent in cfg.ENV.WALKERS:
        with open(f'{ST_folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        ST_scores.append(np.array(log[agent]['reward']['reward'][-10:]).mean())
    order = np.argsort(ST_scores)
    test_agent_id = np.concatenate([order[:20], order[50:350:10], order[-20:]])
    agents = [cfg.ENV.WALKERS[i] for i in test_agent_id]
    ST_scores = np.array(ST_scores)[test_agent_id]

    folders = [
        'output/ft_train_mutate_KL_5_wo_PE+dropout_1409', 
        'output/ft_train_mutate_KL_5_wo_PE+dropout_1410', 
        'output/ft_train_mutate_UED_KL_5_wo_PE+dropout_1409', 
    ]
    names = [
        'maximin_1409', 
        'maximin_1410', 
        'regret_1409', 
    ]

    os.system('mkdir video/output/compare')
    for i, agent in enumerate(agents):
        os.makedirs(f'video/output/compare/{i:03d}_{agent}', exist_ok=True)
        for folder, name in zip(folders, names):
            for video in os.listdir(f'video/{folder}'):
                if agent in video:
                    score = int(video[:-4].split('_')[-1])
                    os.system(f'cp video/{folder}/{video} video/output/compare/{i:03d}_{agent}/{name}_{score}.mp4')
                    break

    plt.figure(figsize=(15, 4))
    for i, folder in enumerate(folders):
        MT_score = []
        for agent in agents:
            for video in os.listdir(f'video/{folder}'):
                if agent in video:
                    score = float(video[:-4].split('_')[-1])
                    MT_score.append(score)
                    break
        plt.subplot(1, 3, i + 1)
        plt.scatter(ST_scores, MT_score)
        plt.title(folder.split('/')[1])
    plt.savefig('video/output/compare_ST_MT.png')



if __name__ == '__main__':

    # os.system('cp -r unimals_100/random_100 unimals_100/random_100_above_1000')
    # agents = [x[:-4] for x in os.listdir('unimals_100/random_100_above_1000/xml')]
    # for agent in agents:
    #     with open(f'output/ST_MLP_random_1000_constant_lr/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     score = np.mean(results[agent]['reward']['reward'][-10:])
    #     if score < 1000:
    #         os.system(f'rm unimals_100/random_100_above_1000/images/{agent}.png')
    #         os.system(f'rm unimals_100/random_100_above_1000/metadata/{agent}.json')
    #         os.system(f'rm unimals_100/random_100_above_1000/unimal_init/{agent}.pkl')
    #         os.system(f'rm unimals_100/random_100_above_1000/xml/{agent}.xml')

    # python tools/video.py --folder output/ft_400M_mutate_400_wo_height_check_env_256_curation_regret_alpha_0.1_KL_5_wo_PE+dropout_rerunb --test_agents data/mjstep_error --suffix 1409 --seed 1409
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--folder", default=None, type=str)
    parser.add_argument("--seed", default=1409, type=int)
    parser.add_argument("--test_agents", default='unimals_100/test', type=str)
    parser.add_argument("--suffix", default=None, type=str)
    args = parser.parse_args()

    seed_folder = f'{args.folder}/{args.seed}'
    if 'modular' not in seed_folder:
        model_path = f'{seed_folder}/Unimal-v0.pt'
    else:
        model_path = f'{seed_folder}/Modular-v0.pt'
    config_path = f'{seed_folder}/config.yaml'
    test_folder = args.test_agents
    if args.suffix is not None:
        output_path = f'video/{args.folder}_{args.suffix}'
    else:
        output_path = f'video/{args.folder}'

    record_video(model_path, config_path, test_folder, output_path)

    # # record videos of zero-shot transfer to test robots
    # folders = dict()
    # folders['walker'] = [
    #     # 'output/modular_walker_train_TF', 
    #     'output/modular_walker_train_mlp', 
    #     # 'output/modular_walker_train_consistentMLP', 
    #     # 'output/modular_walker_train_TF_wo_vecnorm', 
    #     'output/modular_walker_train_mlp_wo_vecnorm', 
    #     # 'output/modular_walker_train_consistentMLP_wo_vecnorm', 
    # ]
    # folders['humanoid'] = [
    #     # 'output/modular_humanoid_train_TF', 
    #     'output/modular_humanoid_train_mlp', 
    #     # 'output/modular_humanoid_train_consistentMLP', 
    #     # 'output/modular_humanoid_train_TF_wo_vecnorm', 
    #     'output/modular_humanoid_train_mlp_wo_vecnorm', 
    #     # 'output/modular_humanoid_train_consistentMLP_wo_vecnorm', 
    # ]
    # # for agent in ['walker', 'humanoid']:
    # #     for folder in folders[agent]:
    # #         for seed in [1409]:
    # #             seed_folder = f'{folder}/{seed}'
    # #             model_path = f'{seed_folder}/Modular-v0.pt'
    # #             config_path = f'{seed_folder}/config.yaml'
    # #             if agent == 'walker':
    # #                 test_folder = 'modular/walkers'
    # #             else:
    # #                 test_folder = 'modular/humanoids'
    # #             # output_path = f'output/video/{agent}/{seed_folder}'
    # #             # print (output_path)
    # #             # record_video(model_path, config_path, test_folder, output_path)
    # #             evaluate_model(model_path, test_folder, seed_folder)

    # for agent in ['walker', 'humanoid']:
    #     for folder in folders[agent]:
    #         for seed in [1409]:
    #             all_robot_obs = []
    #             for robot in os.listdir(f'modular/{agent}_train/xml'):
    #                 folder_name = folder.split('/')[1]
    #                 name = robot.split('.')[0]
    #                 path = f'eval_history/obs_{folder_name}_{seed}_{name}.pkl'
    #                 with open(path, 'rb') as f:
    #                     obs = pickle.load(f)
    #                 obs = obs.reshape([obs.shape[0], obs.shape[1], -1, 19])
    #                 all_robot_obs.append(obs)
    #             all_robot_obs = np.concatenate(all_robot_obs, 0)
    #             plt.figure()
    #             mean = [all_robot_obs[:, :, :, i].ravel().mean() for i in range(19)]
    #             std = [all_robot_obs[:, :, :, i].ravel().std() for i in range(19)]
    #             plt.bar(np.arange(len(mean)), np.array(mean), yerr=std)
    #             plt.savefig(f'eval_history/obs_hist_{folder_name}.png')
    #             plt.close()
