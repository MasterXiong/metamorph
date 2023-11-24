import json
import pickle
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import argparse
from metamorph.config import cfg

from tools.analyze import plot_training_stats


def run():
    # useful_agents = os.listdir('output/log_single_task_wo_pe')
    all_agents = os.listdir('output/log_single_task')
    for agent in all_agents:
        # if agent not in useful_agents:
            # os.system(f'rm -r output/log_single_task/{agent}')
        os.system(f'rm output/log_single_task/{agent}/*.pt')
        os.system(f'rm output/log_single_task/{agent}/*.yaml')
        os.system(f'rm output/log_single_task/{agent}/*.json')
        os.system(f'rm -r output/log_single_task/{agent}/tensorboard')


def simple_ST_training_curve():
    folder = 'output2/ST_MLP_train_mutate_constant_lr'
    train_curve = {}
    for agent in os.listdir(folder):
        with open(f'{folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        train_curve[agent] = log[agent]['reward']['reward']
    
    for agent in train_curve:
        plt.figure()
        plt.plot(train_curve[agent])
        plt.savefig(f'figures/ST_mutate_400/{agent}.png')
        plt.close()


def compare_ST_training(folders, output_folder, prefix, names, all_agents=None):

    all_curves = {}
    for folder in folders:
        train_curve = defaultdict(list)
        if all_agents is None:
            agents = os.listdir(f'output/{folder}')
        else:
            agents = all_agents
        for agent in agents:
            for seed in os.listdir(f'output/{folder}/{agent}'):
                path = f'output/{folder}/{agent}/{seed}'
                if 'checkpoint_-1.pt' in os.listdir(path):
                    with open(f'{path}/Unimal-v0_results.json', 'r') as f:
                        log = json.load(f)
                    train_curve[agent].append(log[agent]['reward']['reward'])
        all_curves[folder] = train_curve

    avg_curve = {}
    for folder in folders:
        avg_curve[folder] = []
    for agent in all_curves[folders[0]]:
        all_include = True
        for folder in folders[1:]:
            if agent not in all_curves[folder]:
                all_include = False
                break
        if not all_include:
            continue
            
        plt.figure()
        for folder in folders:
            c = all_curves[folder][agent]
            length = min([len(x) for x in c])
            all_seeds = np.stack([np.array(x)[:length] for x in c])
            avg, std = all_seeds.mean(0), all_seeds.std(0)
            plt.plot([i*2560*32 for i in range(len(avg))], avg, label=f'{folder} ({all_seeds.shape[0]} seeds)')
            plt.fill_between([i*2560*32 for i in range(len(avg))], avg - std, avg + std, alpha=0.25)
            avg_curve[folder].append(avg)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Returns')
        plt.title(agent)
        plt.savefig(f'figures/{output_folder}/{prefix}_{agent}.png')
        plt.close()
    
    # names = [
    #     'MetaMorph', 
    #     'Node-wise embedding', 
    #     'Node-wise decoder'
    # ]
    agents = list(all_curves[folders[0]].keys())
    agents = [agent for agent in agents if len(all_curves[folders[0]][agent]) >= 2]
    agents = set(agents)
    for folder in folders[1:]:
        new_agents = list(all_curves[folder].keys())
        new_agents = [agent for agent in new_agents if len(all_curves[folder][agent]) >= 2]
        new_agents = set(new_agents)
        agents = agents.intersection(new_agents)
    print (agents)
    plt.figure()
    for i, folder in enumerate(folders):
        seed_all = []
        # for agent in all_curves[folder]:
        #     print (folder, agent, len(all_curves[folder][agent]))
        for j in range(2):
            seed_all.append(np.stack([np.array(all_curves[folder][agent][j]) for agent in agents]).mean(0))
        seed_all = np.stack(seed_all)
        avg = seed_all.mean(0)
        std = seed_all.std(0)
        l = avg.shape[0]
        # l = min([len(x) for x in avg_curve[folder]])
        # avg = np.stack([x[:l] for x in avg_curve[folder]]).mean(axis=0)
        # std = np.stack([x[:l] for x in avg_curve[folder]]).std(axis=0)
        plt.plot([i*2560*32 for i in range(l)], avg, label=names[i])
        plt.fill_between([i*2560*32 for i in range(l)], avg - std, avg + std, alpha=0.25)
    plt.legend(loc='lower right')
    plt.xlabel('Time step')
    plt.ylabel('Returns')
    plt.title(prefix)
    plt.savefig(f'figures/{output_folder}/{prefix}_average.png')
    # plt.savefig(f'figures/motivation/task_average.pdf')
    plt.close()


def compare_train_curve(folders, prefix, agents=['__env__'], seeds=[1409, 1410, 1411], plot_each_seed=False):

    cmap = plt.get_cmap('tab10')
    colors = cmap.colors

    plt.figure()
    for i, name in enumerate(folders.keys()):

        folder = folders[name]
        try:
            cfg.merge_from_file(f'{folder}/1409/config.yaml')
        except:
            cfg.merge_from_file(f'{folder}/1410/config.yaml')

        c = colors[i]
        all_curves = []
        seed_count = 0
        # if '/' in folders[i]:
        #     folder = folders[i]
        # else:
        #     folder = 'output/' + folders[i]
        for seed in seeds:
            if str(seed) not in os.listdir(folder):
                continue
            if 'Unimal-v0_results.json' in os.listdir(f'{folder}/{seed}'):
                with open(f'{folder}/{seed}/Unimal-v0_results.json', 'r') as f:
                    train_results = json.load(f)
            else:
                with open(f'{folder}/{seed}/Modular-v0_results.json', 'r') as f:
                    train_results = json.load(f)
            min_l = min([len(train_results[agent]['reward']['reward']) for agent in agents])
            return_curve = np.stack([train_results[agent]['reward']['reward'][:min_l] for agent in agents]).mean(axis=0)
            if len(return_curve) > 10:
                all_curves.append(return_curve)
                seed_count += 1
        print (seed_count)
        print ([x[-1] for x in all_curves])
        l = min([len(x) for x in all_curves])
        avg_curve = np.stack([x[:l] for x in all_curves])
        timesteps = np.arange(avg_curve.shape[1]) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS
        plt.plot(timesteps, avg_curve.mean(axis=0), c=c, label=f'{name} ({seed_count} seeds)')
        plt.fill_between(timesteps, avg_curve.mean(axis=0) - avg_curve.std(axis=0), avg_curve.mean(axis=0) + avg_curve.std(axis=0), alpha=0.25)
        if plot_each_seed:
            for x in all_curves:
                plt.plot(x[:l], c=c, alpha=0.5)
        print (name, avg_curve.mean(axis=0)[-1])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), ncols=2, prop = {'size':5})
    plt.xlabel('Timesteps')
    plt.ylabel('return')
    plt.savefig(f'figures/train_curve_{prefix}.png')
    plt.close()

    # path = 'output/log_hypernet_1410/checkpoint_500.pt'
    # m, _ = torch.load(path)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # w = m.v_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.subplot(1, 2, 2)
    # w = m.mu_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.savefig('figures/hypernet_weight.png')
    # plt.close()


def compare_train_curves(folders, names, agent='__env__'):

    plt.figure(figsize=(15, 4))
    for n in range(3):
        plt.subplot(1, 3, n + 1)
        for i in range(len(folders[n])):
            all_curves = []
            seed_count = 0
            for seed in [1409, 1410, 1411]:
                try:
                    with open(f'output/{folders[n][i]}/{seed}/Unimal-v0_results.json', 'r') as f:
                        train_results = json.load(f)
                    return_curve = train_results[agent]['reward']['reward']
                    all_curves.append(return_curve)
                    seed_count += 1
                except:
                    pass
            print (seed_count)
            l = min([len(x) for x in all_curves])
            avg_curve = np.stack([x[:l] for x in all_curves])
            plt.plot(avg_curve.mean(axis=0), label=f'{names[i]} ({seed_count} seeds)')
            plt.fill_between(np.arange(avg_curve.shape[1]), avg_curve.mean(axis=0) - avg_curve.std(axis=0), avg_curve.mean(axis=0) + avg_curve.std(axis=0), alpha=0.25)
            print (names[i], avg_curve.mean(axis=0)[-1])
        plt.legend()
        plt.xlabel('PPO update number')
        if n == 0:
            plt.ylabel('return')
    plt.savefig(f'figures/train_curves.png')
    plt.savefig(f'figures/train_curves.pdf')
    plt.close()


def compare_per_robot_score(folders):
    agents = [x[:-4] for x in os.listdir('data/train/xml')]
    plt.figure()
    for name in folders:
        folder = folders[name]
        agent_scores = np.zeros(len(agents))
        for seed in os.listdir(folder):
            with open(f'{folder}/{seed}/Unimal-v0_results.json', 'r') as f:
                results = json.load(f)
            for i, agent in enumerate(agents):
                agent_scores[i] += np.mean(results[agent]['reward']['reward'][-5:])
        agent_scores /= len(os.listdir(folder))
        idx = np.argsort(-agent_scores)
        plt.plot(agent_scores[idx], label=name)
    plt.legend()
    plt.savefig(f'figures/per_robot_train_score.png')
    plt.close()


# def build_unimals_hard():
#     agents = os.listdir('unimals_single_task')
#     scores = {}
#     for agent in agents:
#         scores[agent] = 0.

#     for seed in [1409, 1410]:
#         with open(f'output/ft_baseline/{seed}/Unimal-v0_results.json', 'r') as f:
#             train_results = json.load(f)
#         for agent in agents:
#             scores[agent] += train_results[agent]['reward']['reward'][-1]
    
#     for agent in agents:
#         scores[agent] /= 2.
    
#     agents = list(scores.keys())
#     final_scores = np.array(list(scores.values()))
#     order = np.argsort(final_scores)
#     for i in range(10):
#         print (f'{agents[order[i]]}: {final_scores[order[i]]}')
    
#     os.system('mkdir unimals_hard')
#     os.system('mkdir unimals_hard/xml')
#     os.system('mkdir unimals_hard/metadata')
#     for i in range(10):
#         agent = agents[order[i]]
#         os.system(f'cp unimals_100/train/metadata/{agent}.json unimals_hard/metadata/')
#         os.system(f'cp unimals_100/train/xml/{agent}.xml unimals_hard/xml/')
