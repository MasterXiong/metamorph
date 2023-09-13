import json
import pickle
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import argparse
from metamorph.config import cfg

from analyze import plot_training_stats


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


def compare_train_curve(folders, names, prefix, agents=['__env__'], seeds=[1409, 1410, 1411], plot_each_seed=False):

    cmap = plt.get_cmap('tab10')
    colors = cmap.colors

    plt.figure()
    for i in range(len(folders)):

        try:
            cfg.merge_from_file(f'{folders[i]}/1409/config.yaml')
        except:
            cfg.merge_from_file(f'{folders[i]}/1410/config.yaml')
        folder = folders[i]

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
        plt.plot(timesteps, avg_curve.mean(axis=0), c=c, label=f'{names[i]} ({seed_count} seeds)')
        plt.fill_between(timesteps, avg_curve.mean(axis=0) - avg_curve.std(axis=0), avg_curve.mean(axis=0) + avg_curve.std(axis=0), alpha=0.25)
        if plot_each_seed:
            for x in all_curves:
                plt.plot(x[:l], c=c, alpha=0.5)
        print (names[i], avg_curve.mean(axis=0)[-1])
    plt.legend(prop = {'size':8})
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--stat", action="store_true")
    args = parser.parse_args()

    agents = ['__env__']

    # simple_ST_training_curve()

    folders = [
        'output/ft_expand_base_uniform_sample_KL_5_wo_PE+dropout', 
        'output/ft_expand_base_limb_norm_uniform_sample_KL_5_wo_PE+dropout', 
        'output/ft_random_100_uniform_sample_KL_5_wo_PE+dropout', 
    ]
    names = [
        'expand_base', 
        'expand_base, limb norm', 
        'random_100', 
    ]
    suffix = 'test'
    compare_train_curve(folders, names, suffix, agents=agents, plot_each_seed=False)

    folders = [
        'output/ft_baseline_KL_5_wo_PE+dropout', 
        'output2/ft_MLP_256*3_KL_5_wo_PE+dropout', 
        'output/ft_MLP_separate_IO_uniform_sample_KL_5_wo_PE+dropout', 
        # 'output/ft_MLP_separate_IO_same_init_uniform_sample_KL_5_wo_PE+dropout', 
        # 'ft_400M_train_KL_5_wo_PE+dropout', 
        # 'ft_MLP_separate_IO_new_init_400M_256*3_KL_5_wo_PE+dropout', 
        'output/ft_MLP_400M_separate_IO_uniform_sample_KL_5_wo_PE+dropout', 
        'output/ft_MLP_400M_separate_IO_interchange_update_uniform_sample_KL_5_wo_PE+dropout', 
        'output/ft_MLP_400M_separate_IO+hidden_uniform_sample_KL_5_wo_PE+dropout', 
    ]
    names = [
        'baseline', 
        'MLP', 
        'separate IO, uniform sample', 
        # 'separate IO with same init, uniform sample', 
        # 'baseline, 400M', 
        # 'HN-MLP, new init, 400M', 
        'separate IO, uniform sample, 400M', 
        'separate IO with interchange update, uniform sample, 400M', 
        'separate IO+hidden, uniform sample, 400M', 
    ]
    suffix = 'HN-MLP'
    compare_train_curve(folders, names, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'uniform', 
    #     'PVL', 
    # ]
    # suffix = '400M_mutate_400_curation'
    # compare_train_curve(folders, names, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'uniform', 
    #     'PVL', 
    # ]
    # suffix = '400M_mutate_1000_curation'
    # compare_train_curve(folders, names, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_mutate_400_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_mutate_400_curation_PVL_KL_5_wo_PE+dropout', 
    #     'ft_mutate_400_curation_PVL_staleness_0.3_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'uniform', 
    #     'PVL', 
    #     'PVL, staleness=0.3', 
    # ]
    # suffix = '100M_mutate_400_curation'
    # compare_train_curve(folders, names, suffix, agents=agents, plot_each_seed=False)


    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_tanh_output_std_0.9_KL_5_wo_PE+dropout', 
    #     'ft_tanh_output_std_0.7_KL_5_wo_PE+dropout', 
    #     'ft_tanh_output_std_0.5_KL_5_wo_PE+dropout', 
    #     'ft_tanh_output_std_0.3_KL_5_wo_PE+dropout', 
    #     'ft_tanh_action_KL_5_wo_PE+dropout', 
    # ]
    # suffix = 'tanh'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # for folder in os.listdir('output'):
    #     if folder[0] == '"' and folder[-1] == '"':
    #         print (folder)
    #         os.system(f'mv output/{folder} output/{folder[1:-1]}')
    
    # folders = [
    #     'TF_ST_ft_KL_5_wo_PE+dropout', 
    #     'TF_ST_ft_KL_5_wo_dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with PE', 
    # ]
    # compare_ST_training(folders, 'PE', 'ST_ft', names)

    # folders = [
    #     'TF_ST_incline_KL_5_wo_PE+dropout', 
    #     'TF_ST_incline_KL_5_wo_dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with PE', 
    # ]
    # compare_ST_training(folders, 'PE', 'ST_incline', names)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_PE_zero_init_KL_5_wo_dropout', 
    #     'ft_separate_PE_zero_init_KL_5_wo_PE+dropout', 
    #     'ft_constant_lr_KL_5_wo_PE+dropout', 
    #     'ft_constant_lr_0.0002_KL_5_wo_PE+dropout', 
    #     'ft_constant_lr_0.0001_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with PE, zero init', 
    #     'TF with separate PE, zero init', 
    #     'TF wo PE, constant lr', 
    #     'TF wo PE, constant lr = 0.0002', 
    #     'TF wo PE, constant lr = 0.0001', 
    # ]
    # prefix = 'MT_100_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     'csr_PE_zero_init_KL_3_wo_dropout', 
    #     'csr_separate_PE_zero_init_KL_3_wo_PE+dropout', 
    #     'csr_constant_lr_KL_3_wo_PE+dropout', 
    #     'csr_constant_lr_0.0002_KL_3_wo_PE+dropout', 
    #     'csr_constant_lr_0.0001_KL_3_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with PE, zero init', 
    #     'TF with separate PE, zero init', 
    #     'TF wo PE, constant lr', 
    #     'TF wo PE, constant lr = 0.0002', 
    #     'TF wo PE, constant lr = 0.0001', 
    # ]
    # prefix = 'MT_100_csr'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'incline_baseline_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_KL_5_wo_PE+dropout', 
    #     'incline_decoder_64_PE_zero_init_KL_3_wo_dropout', 
    #     'incline_decoder_64_separate_PE_zero_init_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_constant_lr_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_constant_lr_0.0002_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_constant_lr_0.0001_KL_3_wo_PE+dropout', 
    #     'incline_decoder_64_constant_lr_0.0001_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE, KL=0.03', 
    #     'TF wo PE, KL=0.03, decoder=[64]', 
    #     'TF wo PE, KL=0.05, decoder=[64]', 
    #     'TF wo PE, KL=0.03, decoder=[64], PE with zero init', 
    #     'TF wo PE, KL=0.03, decoder=[64], separate PE with zero init', 
    #     'TF wo PE, KL=0.03, decoder=[64], constant lr', 
    #     'TF wo PE, KL=0.03, decoder=[64], constant lr = 0.0002', 
    #     'TF wo PE, KL=0.03, decoder=[64], constant lr = 0.0001', 
    #     'TF wo PE, KL=0.05, decoder=[64], constant lr = 0.0001', 
    # ]
    # prefix = 'MT_100_incline'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'ft_fine_tune_full_model_KL_5_wo_PE+dropout', 
    #     'ft_fine_tune_PE_KL_5_wo_dropout', 
    #     'ft_fine_tune_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'tune the whole model', 
    #     'tune vanilla PE', 
    #     'tune separate PE', 
    # ]
    # prefix = 'fine_tune_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_separate_PE_zero_init_600_KL_5_wo_PE+dropout', 
    #     'ft_separate_PE_zero_init_300_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF', 
    #     'TF with separate PE after 600 iters', 
    #     'TF with separate PE after 300 iters', 
    # ]
    # prefix = 'ft_separate_PE'
    # compare_train_curve(folders, names, prefix)

    # check different obs normalization methods
    # names = [
    #     'VecNorm (default)', 
    #     'limb-level VecNorm', 
    #     'limb-level VecNorm, excluding zero-padding', 
    #     'fix-range norm for context; no norm for state', 
    #     'fix-range norm for context and state', 
    # ]

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_limb_norm_KL_5_wo_PE+dropout', 
    #     'ft_limb_norm_wo_padding_KL_5_wo_PE+dropout', 
    #     'ft_fixed_range_context_KL_5_wo_PE+dropout', 
    #     'ft_fixed_range_state+context_KL_5_wo_PE+dropout', 
    # ]
    # prefix = 'norm_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     'csr_limb_norm_KL_3_wo_PE+dropout', 
    #     'csr_limb_norm_wo_padding_KL_3_wo_PE+dropout', 
    #     'csr_fixed_range_context_KL_3_wo_PE+dropout', 
    #     'csr_fixed_range_state+context_KL_3_wo_PE+dropout', 
    # ]
    # prefix = 'norm_csr'
    # compare_train_curve(folders, names, prefix)

    # names = [
    #     'VecNorm (default)', 
    #     'limb-level VecNorm', 
    #     # 'limb-level VecNorm, excluding zero-padding', 
    #     # 'fix-range norm for context; no norm for state', 
    #     'no normalization', 
    #     'fixed normalization', 
    # ]

    # folders = [
    #     'modular_humanoid_train_TF', 
    #     'modular_humanoid_train_TF_limb_norm', 
    #     'modular_humanoid_train_TF_wo_vecnorm', 
    #     'modular_humanoid_train_TF_fixed_norm', 
    # ]
    # prefix = 'norm_modular_humanoid'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'modular_walker_train_TF', 
    #     'modular_walker_train_TF_limb_norm', 
    #     'modular_walker_train_TF_wo_vecnorm', 
    #     'modular_walker_train_TF_fixed_norm', 
    # ]
    # prefix = 'norm_modular_walker'
    # compare_train_curve(folders, names, prefix)

    names = [
        'maximin', 
        'uniform task sampling', 
        'UED', 
        'maximin recheck', 
        'UED recheck', 
        'ACCEL (train on mutate_400)', 
        'ACCEL (train on default_100)', 
        'metamorph (train on default_100)', 
        # 'ModuMorph', 
    ]
    # folders = [
    #     'ft_train+joint_angle_300_KL_5_wo_PE+dropout', 
    #     'ft_train+joint_angle_300_uniform_task_KL_5_wo_PE+dropout', 
    #     'ft_train+joint_angle_300_UED_KL_5_wo_PE+dropout', 
    #     'ft_train+joint_angle_300_HN+FA_KL_5_wo_PE+dropout', 
    # ]
    # prefix = 'train+joint_angle_300'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'ft_train_mutate_KL_5_wo_PE+dropout', 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout', 
    #     'ft_train_mutate_recheck_KL_5_wo_PE+dropout', 
    #     'ft_train_mutate_UED_recheck_KL_5_wo_PE+dropout', 
    #     # 'ft_train_mutate_HN+FA_KL_5_wo_PE+dropout', 
    #     'output/ft_debug_train_mutate_accel_wo_generation_KL_5_wo_PE+dropout', 
    #     'output/ft_debug_train_accel_wo_generation_KL_5_wo_PE+dropout', 
    #     'output/ft_baseline_KL_5_wo_PE+dropout', 
    # ]
    # prefix = 'train_mutate'
    # agents = [x[:-4] for x in os.listdir('unimals_100/train/xml')]
    # compare_train_curve(folders, names, prefix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_env_64_uniform_sample_KL_5_wo_PE+dropout', 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_uniform_sample_KL_5_wo_PE+dropout', 
    #     # 'ft_400M_mutate_400_env_64_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout', 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_env_128_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '400M_check_env_num'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_baseline_env_64_timestep_1280_KL_5_wo_PE+dropout', 
    #     'ft_baseline_env_128_timestep_640_KL_5_wo_PE+dropout', 
    #     'ft_baseline_env_64_KL_5_wo_PE+dropout', 
    #     'ft_baseline_env_128_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '100M_check_env_num'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_curation_LP_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_curation_LP_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '400M_LP_env_32'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_400_env_256_curation_LP_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_env_256_curation_LP_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '400M_LP_env_256'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_FA_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '400M_random_1000_HN+FA'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # folders = [
    #     'ft_400M_train_KL_5_wo_PE+dropout', 
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_separatePE_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # suffix = '400M_separatePE'
    # compare_train_curve(folders, folders, suffix, agents=agents, plot_each_seed=False)

    # names = [
    #     'default 100 (100M)', 
    #     'default 100 (400M)', 
    #     'train+joint_angle_300 (400M)', 
    #     'train+joint_angle_random (400M)', 
    # ]
    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_400M_train_KL_5_wo_PE+dropout', 
    #     'ft_400M_train+joint_angle_300_KL_5_wo_PE+dropout',  
    #     'ft_400M_train+joint_angle_random_KL_5_wo_PE+dropout', 
    # ]
    # prefix = '400M_joint_angle'
    # compare_train_curve(folders, names, prefix, seeds=[1409])

    # source_dir = 'output/ST_MLP_train_mutate_constant_lr'
    # target_folder = 'output/ST_MLP_joint_angle_300_constant_lr'
    # for agent in os.listdir(source_dir):
    #     if 'mutate' not in agent:
    #         os.system(f'cp -r {source_dir}/{agent} {target_folder}/')

    # folder = 'unimals_100/train_mutate_v2/metadata'
    # max_limb_num = 0
    # for agent in os.listdir(folder):
    #     with open(f'{folder}/{agent}', 'r') as f:
    #         metadata = json.load(f)
    #     max_limb_num = max(max_limb_num, metadata['dof'])
    # print (max_limb_num)

    # names = [
    #     'no PE', 
    #     'vanilla PE', 
    #     'semantic PE', 
    #     'vanilla PE before scaling', 
    #     'semantic PE before scaling', 
    # ]

    # folders = [
    #     'modular_humanoid_train_TF_wo_vecnorm', 
    #     'modular_humanoid_train_TF_wo_vecnorm_vanilla_PE', 
    #     'modular_humanoid_train_TF_wo_vecnorm_semantic_PE', 
    #     'modular_humanoid_train_TF_wo_vecnorm_vanilla_PE_before_scaling', 
    #     'modular_humanoid_train_TF_wo_vecnorm_semantic_PE_before_scaling', 
    # ]
    # prefix = 'PE_modular_humanoid'
    # compare_train_curve(folders, names, prefix)

    # names = [
    #     'no PE', 
    #     # 'vanilla PE', 
    #     # 'semantic PE', 
    #     'vanilla PE before scaling', 
    #     'semantic PE before scaling', 
    # ]

    # folders = [
    #     'modular_humanoid_train_TF_limb_norm', 
    #     'modular_humanoid_train_TF_limb_norm_vanilla_PE', 
    #     'modular_humanoid_train_TF_limb_norm_semantic_PE', 
    # ]
    # prefix = 'PE_modular_humanoid_limb_norm'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_10_ft_KL_5_wo_PE+dropout', 
    #     'unimal_10_ft_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_10_random_env_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_10_incline_KL_5_wo_PE+dropout', 
    #     'unimal_10_incline_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_10_random_env_incline'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_20_ft_fix_env_KL_5_wo_PE+dropout', 
    #     'unimal_20_ft_fix_env_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_20_all_env_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_20_incline_fix_env_KL_5_wo_PE+dropout', 
    #     'unimal_20_incline_fix_env_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_20_all_env_incline'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_10_ft_fix_env_KL_5_wo_PE+dropout', 
    #     'unimal_10_ft_fix_env_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_10_all_env_ft'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'unimal_10_incline_fix_env_KL_5_wo_PE+dropout', 
    #     'unimal_10_incline_fix_env_separate_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'TF wo PE', 
    #     'TF with separate PE'
    # ]
    # prefix = 'PE/MT_10_all_env_incline'
    # compare_train_curve(folders, names, prefix)

    # folders = [
    #     'ST_baseline_KL_5_wo_PE+dropout', 
    #     'ST_per_node_embed_KL_5_wo_PE+dropout', 
    #     'ST_per_node_decoder_KL_5_wo_PE+dropout', 
    # ]
    # compare_ST_training(folders)

    # folders = [
    #     # 'MLP_ST_64*2_KL_5', 
    #     'MLP_ST_128*2_KL_5', 
    #     'MLP_ST_256*2_KL_5', 
    #     'MLP_ST_512*2_KL_5', 
    #     'ST_baseline_KL_5_wo_PE+dropout', 
    # ]
    # compare_ST_training(folders, 'mlp', 'hidden_dim')

    # folders = [
    #     # 'MLP_ST_128*1_KL_5', 
    #     'MLP_ST_128*2_KL_5', 
    #     'MLP_ST_128*3_KL_5', 
    #     'MLP_ST_128*4_KL_5', 
    #     'ST_baseline_KL_5_wo_PE+dropout', 
    # ]
    # compare_ST_training(folders, 'mlp', 'layer_num')

    # compare_ST_training(['MLP_ST_exploration_256*3_KL_3'], 'mlp', 'exploration', ['MLP'])

    # all_folders = []
    # names = [
    #     # 'MLP old', 
    #     'Transformers', 
    #     'MLP', 
    #     'MLP, input with shared init', 
    #     # 'MLP with HN', 
    #     # 'MLP with HN mean', 
    #     # 'MLP with HN, separate context encoder for input and output', 
    #     # 'MLP with HN, new init + single value', 
    #     # 'MLP with HN, new init', 
    #     # 'MLP with HN for input', 
    #     # 'MLP with HN for output', 
    #     # 'MLP with shared input, relu before mean', 
    #     # 'MLP with shared input, relu after mean', 
    #     # 'MLP with HN for input, context input init=0.01', 
    #     # 'MLP with HN for input, nonzero weight init for HN output layer', 
    #     # 'MLP, excluding zero-padding limbs', 
    #     # 'MLP, HN for input, excluding zero-padding limbs, sum', 
    #     # 'MLP, HN for input, excluding zero-padding limbs, mean', 
    #     # 'MLP, HN for input and output, excluding zero-padding limbs, sum', 
    #     # 'MLP, HN for input, smaller lr for HN', 
    #     'MLP, HN for input, relu + sum', 
    #     # 'MLP, HN for input, relu before scaled sum', 
    #     'MLP, HN for input, sum + relu', 
    #     # 'MLP, HN for input, relu + sum + BN', 
    #     # 'MLP, HN for input, relu + mean', 
    #     # 'MLP, HN for input, relu + sum, KL=0.07', 
    #     # 'MLP, HN for input, relu + sum, KL=0.1', 
    #     # 'MLP, HN for input and output, relu + sum', 
    #     'MLP, HN for input, relu + sum, squash = 1.0', 
    #     # 'MLP, HN for input, relu + sum, anneal', 
    #     'MLP, HN for input, relu + sum, squash = 0.2', 
    #     'MLP, HN for input, relu + sum, squash = 0.5', 
    #     # 'MLP, HN for input, sum + relu, squash', 
    # ]
    # stat_idx = [0,1,2,3,4]

    # names = [
    #     'Transformer', 
    #     'MLP', 
    # ]

    # folders = [
    #     # 'ft_MLP_old_256*3_KL_5_wo_PE+dropout', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_256*3_KL_5_wo_PE+dropout', 
    #     'ft_MLP_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     'ft_MLP_input_shared_init_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_mean_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_separate_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_new_init_single_value_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_output_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_shared_input_relu_before_mean_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_shared_input_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_context_input_init_0.01_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_nonzero_weight_init_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_exclude_zero_padding_mean_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     # 'ft_MLP_HN_only_input_small_lr_256*3_KL_5', 
    #     'ft_MLP_HN_only_input_relu_before_agg_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_relu_before_agg_scale_256*3_KL_5', 
    #     'ft_MLP_HN_only_input_relu_after_agg_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_relu-sum-BN_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_relu-mean_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_relu-sum_256*3_KL_7', 
    #     # 'ft_MLP_HN_only_input_relu-sum_256*3_KL_10', 
    #     # 'ft_MLP_HN_input+output_relu-sum_256*3_KL_5', 
    #     'ft_MLP_HN_only_input_relu-sum_squash_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_relu-sum_anneal_256*3_KL_5', 
    #     'ft_MLP_HN_only_input_relu-sum_squash_0.2_256*3_KL_5', 
    #     'ft_MLP_HN_only_input_relu-sum_squash_0.5_256*3_KL_5', 
    #     # 'ft_MLP_HN_only_input_sum-relu_squash_256*3_KL_5', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_architecture'
    # compare_train_curve(folders, names, suffix)
    # # all_folders.append(folders)
    # kl = [0.05 for _ in folders]
    # if args.stat:
    #     for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    #         selected_folders = [folders[i] for i in stat_idx]
    #         selected_names = [names[i] for i in stat_idx]
    #         plot_training_stats(selected_folders, selected_names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     # 'incline_MLP_old_256*3_KL_5_wo_PE+dropout', 
    #     'incline_baseline_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_256*3_KL_5_wo_PE+dropout', 
    #     'incline_MLP_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     'incline_MLP_input_shared_init_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_mean_HN_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_separate_HN_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_new_init_single_value_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_output_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_shared_input_relu_before_mean_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_shared_input_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_context_input_init_0.01_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_nonzero_weight_init_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_exclude_zero_padding_mean_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_exclude_zero_padding_256*3_KL_5_wo_PE+dropout', 
    #     # 'incline_MLP_HN_only_input_small_lr_256*3_KL_5', 
    #     'incline_MLP_HN_only_input_relu_before_agg_256*3_KL_5', 
    #     # 'incline_MLP_HN_only_input_relu_before_agg_scale_256*3_KL_5', 
    #     'incline_MLP_HN_only_input_relu_after_agg_256*3_KL_5', 
    #     # 'incline_MLP_HN_only_input_relu-sum-BN_256*3_KL_5', 
    #     # 'incline_MLP_HN_only_input_relu-mean_256*3_KL_5', 
    #     # 'incline_MLP_HN_input+output_relu-sum_256*3_KL_5', 
    #     'incline_MLP_HN_only_input_relu-sum_squash_256*3_KL_5', 
    #     # 'incline_MLP_HN_only_input_relu-sum_anneal_256*3_KL_5', 
    # ]
    # names = folders
    # suffix = prefix = 'incline_architecture'
    # compare_train_curve(folders, names, suffix)
    # # all_folders.append(folders)
    # kl = [0.05 for _ in folders]
    # if args.stat:
    #     for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    #         selected_folders = [folders[i] for i in stat_idx]
    #         selected_names = [names[i] for i in stat_idx]
    #         plot_training_stats(selected_folders, selected_names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     # 'csr_MLP_old_256*3_KL_3_wo_PE+dropout', 
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_256*3_KL_3_wo_PE+dropout', 
    #     'csr_MLP_exclude_zero_padding_256*3_KL_3_wo_PE+dropout', 
    #     'csr_MLP_input_shared_init_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_mean_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_separate_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_new_init_single_value_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_new_init_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_output_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_shared_input_relu_before_mean_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_shared_input_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_context_input_init_0.01_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_nonzero_weight_init_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_exclude_zero_padding_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_exclude_zero_padding_mean_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_exclude_zero_padding_256*3_KL_3_wo_PE+dropout', 
    #     # 'csr_MLP_HN_only_input_small_lr_256*3_KL_3', 
    #     'csr_MLP_HN_only_input_relu_before_agg_256*3_KL_3', 
    #     # 'csr_MLP_HN_only_input_relu_before_agg_scale_256*3_KL_3', 
    #     'csr_MLP_HN_only_input_relu_after_agg_256*3_KL_3', 
    # ]
    # names = folders
    # suffix = prefix = 'csr_architecture'
    # compare_train_curve(folders, names, suffix)
    # # all_folders.append(folders)
    # kl = [0.03 for _ in folders]
    # if args.stat:
    #     for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    #         selected_folders = [folders[i] for i in stat_idx if i < len(folders)]
    #         selected_names = [names[i] for i in stat_idx if i < len(names)]
    #         plot_training_stats(selected_folders, selected_names, key=stat, prefix=prefix, kl_threshold=kl)

    # compare_train_curves(all_folders, names)

    # folders = [
    #     # 'ft_baseline', 
    #     # 'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_FA_KL_5_wo_PE+dropout', 
    #     'ft_FA_state_input_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA_wo_additional_context_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_ablation'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     # 'ft_baseline', 
    #     # 'ft_baseline_KL_5_wo_PE+dropout', 
    #     'incline_FA_KL_5_wo_PE+dropout', 
    #     'incline_FA_state_input_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_KL_5_wo_PE+dropout', 
    #     'incline_HN+FA_KL_5_wo_PE+dropout', 
    #     'incline_HN+FA_wo_additional_context_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'incline_ablation'
    # compare_train_curve(folders, names, suffix)

    # context PE with different variants
    # folders = [
    #     'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_graph_PE_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_tree_PE_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_tree+graph_PE_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_all_context_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_all_context+connectivity_KL_5_wo_PE+dropout', 
    # ]
    # folders = [
    #     'ft_context+depth_PE_rnn+tf_KL_5_wo_PE+dropout', 
    #     'ft_context_PE_all_KL_5_wo_PE+dropout', 
    #     'ft_context_PE_graph_PE_KL_5_wo_PE+dropout', 
    #     'ft_context_PE_graph_PE_scaled_KL_5_wo_PE+dropout', 
    #     'ft_context_PE_tree_PE_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft'
    # compare_train_curve(folders, names, suffix)
    # kl = [0.05 for _ in folders]
    # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_baseline_KL_5_wo_dropout', 
    #     'ft_context+depth_PE_rnn+tf_KL_5_wo_PE+dropout', 
    #     'ft_context+depth_PE_rnn+tf_scaled_KL_5_wo_PE+dropout', 
    #     'ft_context+depth_PE_rnn+tf_less_state_KL_5_wo_PE+dropout', 
    #     'ft_context+depth_PE_rnn+tf_scaled_less_state_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_PE'
    # compare_train_curve(folders, names, suffix)
    # kl = [0.05 for _ in folders]
    # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'ft_baseline', 
    #     'ft_baseline_KL_5_wo_dropout', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     # 'ft_fix_attention_rnn+tf_KL_5_wo_PE+dropout', 
    #     # 'ft_fix_attention_rnn+tf_scaled_KL_5_wo_PE+dropout', 
    #     # 'ft_fix_attention_rnn+tf_scaled_less_state_KL_5_wo_PE+dropout', 
    #     # 'ft_fix_attention_MLP_morphology_attn_KL_5_wo_PE+dropout', 
    #     'ft_fix_attention_MLP_morphology_attn_3_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_depth_in_context_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_wo_scaling_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_FA'
    # compare_train_curve(folders, names, suffix)
    # # kl = [0.05 for _ in folders]
    # # kl[0] = 0.2
    # # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    # #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # # folders = [
    # #     'obstacle_baseline_KL_3_wo_dropout', 
    # #     'obstacle_context+depth_PE_rnn+tf_KL_3_wo_PE+dropout', 
    # #     'obstacle_context+depth_PE_rnn+tf_scaled_KL_3_wo_PE+dropout', 
    # #     'obstacle_context+depth_PE_rnn+tf_scaled_less_state_KL_3_wo_PE+dropout', 
    # # ]
    # # names = folders
    # # suffix = prefix = 'obstacle_PE'
    # # compare_train_curve(folders, names, suffix)
    # # kl = [0.03 for _ in folders]
    # # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    # #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'obstacle_baseline', 
    #     'obstacle_baseline_KL_3_wo_dropout', 
    #     'obstacle_baseline_KL_3_wo_PE+dropout', 
    #     'obstacle_MLP_fix_attention_KL_3_wo_dropout', 
    #     # 'obstacle_fix_attention_rnn+tf_KL_3_wo_PE+dropout', 
    #     # 'obstacle_fix_attention_rnn+tf_scaled_KL_3_wo_PE+dropout', 
    #     # 'obstacle_fix_attention_rnn+tf_sclad_less_state_KL_3_wo_PE+dropout', 
    #     'obstacle_fix_attention_MLP_morphology_attn_3_KL_3_wo_PE+dropout', 
    #     # 'obstacle_HN_wo_scaling_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'obstacle_FA'
    # compare_train_curve(folders, names, suffix)
    # # kl = [0.03 for _ in folders]
    # # kl[0] = 0.2
    # # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    # #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # # folders = [
    # #     'csr_baseline_KL_3_wo_dropout', 
    # #     'csr_context+depth_PE_rnn+tf_KL_3_wo_PE+dropout', 
    # #     'csr_context+depth_PE_rnn+tf_scaled_KL_3_wo_PE+dropout', 
    # #     'csr_context+depth_PE_rnn+tf_scaled_lss_state_KL_3_wo_PE+dropout', 
    # # ]
    # # names = folders
    # # suffix = prefix = 'csr_PE'
    # # compare_train_curve(folders, names, suffix)
    # # kl = [0.03 for _ in folders]
    # # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    # #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'csr_baseline', 
    #     'csr_baseline_KL_3_wo_dropout', 
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     'csr_fix_attention_MLP_KL_3_wo_PE+dropout', 
    #     # 'csr_fix_attention_rnn+tf_KL_3_wo_PE+dropout', 
    #     # 'csr_fix_attention_rnn+tf_scaled_KL_3_wo_PE+dropout', 
    #     # 'csr_fix_attention_rnn+tf_scaled_less_state_KL_3_wo_PE+dropout', 
    #     # 'csr_fix_attention_MLP_morphology_attn_KL_5_wo_PE+dropout', 
    #     'csr_fix_attention_MLP_morphology_3_attn_KL_3_wo_PE+dropout', 
    #     # 'csr_HN_depth_in_context_KL_3_wo_PE+dropout', 
    #     # 'csr_HN_wo_scaling_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'csr_FA'
    # compare_train_curve(folders, names, suffix)
    # # kl = [0.03 for _ in folders]
    # # kl[0] = 0.2
    # # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    # #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'output/modular_hopper_mlp', 
    #     'output/modular_hopper_consistentMLP', 
    #     'output/modular_hopper_mlp_wo_vecnorm', 
    #     'output/modular_hopper_consistentMLP_wo_vecnorm', 
    #     'output/modular_hopper_TF', 
    # ]
    # names = folders
    # suffix = 'modular_hopper'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_walker_mlp', 
    #     'output/modular_walker_consistentMLP', 
    #     'output/modular_walker_mlp_wo_vecnorm', 
    #     'output/modular_walker_consistentMLP_wo_vecnorm', 
    #     'output/modular_walker_TF', 
    # ]
    # names = folders
    # suffix = 'modular_walker'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_humanoid_mlp', 
    #     'output/modular_humanoid_consistentMLP', 
    #     'output/modular_humanoid_mlp_wo_vecnorm', 
    #     'output/modular_humanoid_consistentMLP_wo_vecnorm', 
    #     'output/modular_humanoid_TF', 
    # ]
    # names = folders
    # suffix = 'modular_humanoid'
    # compare_train_curve(folders, names, suffix)

    # names = [
    #     'TF with VecNorm', 
    #     'MLP with VecNorm', 
    #     'CMLP with VecNorm', 
    #     'TF wo VecNorm', 
    #     'MLP wo VecNorm', 
    #     'CMLP wo VecNorm', 
    # ]

    # folders = [
    #     'output/modular_walker_train_TF', 
    #     'output/modular_walker_train_mlp', 
    #     'output/modular_walker_train_consistentMLP', 
    #     'output/modular_walker_train_TF_wo_vecnorm', 
    #     'output/modular_walker_train_mlp_wo_vecnorm', 
    #     'output/modular_walker_train_consistentMLP_wo_vecnorm', 
    # ]
    # # names = folders
    # suffix = 'modular_walker_train'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_humanoid_train_TF', 
    #     'output/modular_humanoid_train_mlp', 
    #     'output/modular_humanoid_train_consistentMLP', 
    #     'output/modular_humanoid_train_TF_wo_vecnorm', 
    #     'output/modular_humanoid_train_mlp_wo_vecnorm', 
    #     'output/modular_humanoid_train_consistentMLP_wo_vecnorm', 
    # ]
    # # names = folders
    # suffix = 'modular_humanoid_train'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_all_train_TF', 
    #     # 'output/modular_all_train_TF_KL_10', 
    #     'output/modular_all_train_mlp', 
    #     'output/modular_all_train_consistentMLP', 
    # ]
    # names = folders
    # suffix = 'modular_all_train'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_hopper_full_mlp', 
    # ]
    # names = folders
    # suffix = 'modular_hopper_full'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_walker_full_mlp', 
    # ]
    # names = folders
    # suffix = 'modular_walker_full'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'output/modular_humanoid_full_mlp', 
    # ]
    # names = folders
    # suffix = 'modular_humanoid_full'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'best_models/ft_baseline_KL_5_wo_PE+dropout', 
    #     'best_models/ft_HN+FA_KL_5_wo_PE+dropout', 
    #     'best_models/ft_FA_KL_5_wo_PE+dropout', 
    #     # 'ablation/ft_baseline_SWAT_PE_KL_5_wo_PE+dropout', 
    #     'ablation/ft_baseline_SWAT_RE_KL_5_wo_PE+dropout', 
    #     'ablation/ft_FA+SWAT_RE_KL_5_wo_PE+dropout', 
    #     # 'ablation/ft_HN+FA_wo_context_input_KL_5_wo_PE+dropout', 
    #     'ablation/ft_HN_input+FA_KL_5_wo_PE+dropout', 
    #     'ablation/ft_HN_output+FA_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_ablation'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'best_models/incline_baseline_KL_3_wo_PE+dropout', 
    #     'best_models/incline_HN+FA_KL_5_wo_PE+dropout', 
    #     'ablation/incline_HN+FA_wo_context_input_KL_5_wo_PE+dropout', 
    #     'ablation/incline_baseline_SWAT_PE_KL_3_wo_PE+dropout', 
    #     'ablation/incline_baseline_SWAT_RE_KL_3_wo_PE+dropout', 
    #     'ablation/incline_HN_input+FA_KL_5_wo_PE+dropout', 
    #     'ablation/incline_HN_output+FA_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'incline_ablation'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'best_models/exploration_baseline_KL_3_wo_PE+dropout', 
    #     'best_models/exploration_HN+FA_KL_3_wo_PE+dropout', 
    #     'ablation/exploration_HN+FA_wo_context_input_KL_3_wo_PE+dropout', 
    #     'ablation/exploration_HN_input+FA_KL_3_wo_PE+dropout', 
    #     'ablation/exploration_HN_output+FA_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'exploration_ablation'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'best_models/csr_200M_baseline_KL_3_wo_PE+dropout', 
    #     'best_models/csr_200M_HN+FA_KL_3_wo_PE+dropout', 
    #     'ablation/csr_HN+FA_wo_context_input_KL_3_wo_PE+dropout', 
    #     'ablation/csr_200M_baseline_SWAT_RE_KL_3_wo_PE+dropout', 
    #     'ablation/csr_200M_HN_input+FA_KL_3_wo_PE+dropout', 
    #     'ablation/csr_200M_HN_output+FA_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'csr_ablation'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'best_models/obstacle_200M_baseline_KL_3_wo_PE+dropout', 
    #     'best_models/obstacle_200M_HN+FA_KL_3_wo_PE+dropout', 
    #     'ablation/obstacle_HN+FA_wo_context_input_KL_3_wo_PE+dropout', 
    #     'ablation/obstacle_200M_HN_input+FA_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'obstacle_ablation'
    # compare_train_curve(folders, names, suffix)

    # kl = [0.05 for _ in folders]
    # kl[0] = 0.2
    # kl[4] = 0.2
    # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'surr1', 'surr2', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'ft_baseline', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_baseline_KL_5_wo_embed_scaling_wo_dropout', 
    #     'ft_HN_depth_in_context_KL_5_wo_PE+dropout', 
    #     'ft_HN_wo_scaling_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_wo_scaling_wo_PE', 
    #     # 'ft_HN_wo_scaling_init_0.01_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_embed_only_wo_scaling_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_decoder_only_wo_scaling_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA*2_wo_scaling_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_wo_scaling_MLP*2_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_HN'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'ft_baseline', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     # 'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_FA_KL_5_wo_PE+dropout', 
    #     'ft_HN_KL_5_wo_PE+dropout', 
    #     # 'ft_HN_KL_3_wo_PE+dropout', 
    #     # 'ft_HN_depth+childnum_context_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA_KL_5_wo_PE+dropout', 
    #     # 'ft_HN+FA_KL_3_wo_PE+dropout', 
    #     # 'ft_HN_depth+childnum_context_embed_only_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_paper'
    # compare_train_curve(folders, names, suffix)
    # kl = [0.2, 0.05, 0.05, 0.05, 0.05]
    # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_HN_depth+childnum_context_KL_5_wo_PE+dropout', 
    #     'ft_HN+FA_KL_5_wo_PE+dropout', 
    #     'ft_HN_depth_context_KL_5_wo_PE+dropout', 
    #     'ft_HN_depth+tree_context_KL_5_wo_PE+dropout', 
    #     'ft_HN_depth_context_embed_only_KL_5_wo_PE+dropout', 
    #     'ft_HN_depth_context_decoder_only_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_HN'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     # 'ft_baseline_KL_5_per_node_embed_wo_PE+dropout', 
    #     # 'ft_baseline_KL_5_per_node_decoder_wo_PE+dropout', 
    #     # 'ft_per_node_decoder_KL_3_wo_PE+dropout', 
    #     'ft_per_node_decoder_KL_5_wo_PE+dropout', 
    #     # 'ft_per_node_embed_KL_3_wo_PE+dropout', 
    #     # 'ft_per_node_embed_KL_5_wo_PE+dropout', 
    #     'ft_per_node_embed', 
    #     'ft_per_node_decoder', 
    # ]
    # # folders = [
    # #     'ft_hard_baseline_KL_5_wo_PE+dropout', 
    # #     'ft_hard_per_node_embed_KL_5_wo_PE+dropout', 
    # # ]
    # names = folders
    # suffix = prefix = 'ft_per_node_param'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     'csr_HN+FA_KL_3_wo_PE+dropout', 
    # ]
    # agents = [
    #     'floor-1409-8-8-01-14-47-11', 
    #     'vt-5506-9-3-02-15-19-17', 
    #     'floor-1409-11-8-01-08-25-42', 
    #     'vt-1409-12-10-02-21-04-52', 
    #     'floor-1409-6-12-01-13-27-36', 
    #     'floor-1409-1-4-01-09-49-50', 
    #     'vt-1409-15-5-02-21-44-24', 
    # ]
    # names = folders
    # for agent in agents:
    #     print (agent)
    #     suffix = f'csr_{agent}'
    #     compare_train_curve(folders, names, suffix, agent=agent)

    # folders = [
    #     'ft_baseline', 
    #     'ft_HN_wo_PE', 
    #     'ft_FA_wo_PE', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_dropout'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'csr_baseline', 
    #     'csr_baseline_KL_3_wo_PE+dropout', 
    #     'csr_FA_new_KL_3_wo_PE+dropout', 
    #     'csr_FA_hfield_KL_3_wo_PE+dropout'
    # ]
    # names = folders
    # suffix = prefix = 'csr'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'obstacle_baseline', 
    #     'obstacle_baseline_KL_3_wo_PE+dropout', 
    #     'obstacle_FA_new_KL_3_wo_PE+dropout', 
    #     'obstacle_FA_hfield_KL_3_wo_PE+dropout'
    # ]
    # names = folders
    # suffix = prefix = 'obstacle'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'v2_ft_baseline_KL_5_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'ft_check'
    # compare_train_curve(folders, names, suffix)

    # folders = [
    #     'incline_baseline', 
    #     # 'incline_baseline_KL_5_wo_PE+dropout', 
    #     'incline_baseline_KL_3_wo_PE+dropout', 
    #     # 'incline_MLP_fix_attention_KL_5_wo_dropout', 
    #     'incline_FA_KL_5_wo_PE+dropout', 
    #     # 'incline_FA_KL_3_wo_PE+dropout', 
    #     'incline_HN_KL_5_wo_PE+dropout', 
    #     # 'incline_HN_KL_3_wo_PE+dropout', 
    #     'incline_HN+FA_KL_5_wo_PE+dropout', 
    #     'incline_HN+FA_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'incline'
    # compare_train_curve(folders, names, suffix)
    # # kl = [0.05, 0.03, 0.05, 0.03, 0.05, 0.03, 0.05]
    # kl = [0.05, 0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03]
    # for stat in ['approx_kl', 'grad_norm', 'ratio', 'val_loss', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)

    # folders = [
    #     'incline_baseline_decoder*2_KL_3_wo_PE+dropout', 
    #     'incline_baseline_decoder*2_KL_5_wo_PE+dropout', 
    #     'incline_FA_decoder*2_KL_5_wo_PE+dropout', 
    #     'incline_FA_decoder*2_KL_3_wo_PE+dropout', 
    #     'incline_HN_decoder*2_KL_5_wo_PE+dropout', 
    #     'incline_HN_decoder*2_KL_3_wo_PE+dropout', 
    #     'incline_HN+FA_decoder*2_KL_5_wo_PE+dropout', 
    #     'incline_HN+FA_decoder*2_KL_3_wo_PE+dropout', 
    # ]
    # names = folders
    # suffix = prefix = 'incline_decoder*2'
    # compare_train_curve(folders, names, suffix)
    # kl = [0.05, 0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03]
    # for stat in ['approx_kl', 'grad_norm', 'ratio', 'val_loss', 'clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)