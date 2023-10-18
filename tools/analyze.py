# python tools/evaluate.py --policy_path log_origin --policy_name Unimal-v0
import argparse
import os
import torch
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import copy
import math

from metamorph.config import cfg

from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.algos.ppo.model import Agent

from collections import defaultdict

from tools.evaluate import evaluate_model

torch.manual_seed(0)


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


# plot histgram of a context feature
def plot_context_hist(feature):

    cfg.ENV.WALKER_DIR = 'unimals_100/train'
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = './eval'
    cfg.MODEL.CONTEXT_OBS_TYPES = [feature]
    cfg.PPO.NUM_ENVS = 2
    set_cfg_options()

    agents = list(os.listdir('unimals_single_task'))
    context = []
    print (feature)
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        obs = env.reset()
        context.append(obs['context'].reshape(cfg.MODEL.MAX_LIMBS, -1)[:env.metadata["num_limbs"]])
        env.close()
    context = np.concatenate(context, axis=0)

    print (context.min(axis=0))
    print (context.max(axis=0))

    # for i in range(context.shape[1]):
    #     plt.figure()
    #     plt.hist(context[:, i], bins=100)
    #     plt.title(f'{feature}_{i}: min: {context[:, i].min()}, max: {context[:, i].max()}')
    #     plt.savefig(f'figures/context_hist/{feature}_{i}.png')
    #     plt.close()


def plot_training_stats(folders, names=None, key='grad_norm', batch_size=None, env_num=None, kl_threshold=None, prefix=None):
    os.makedirs('figures/train_stats', exist_ok=True)
    if names is None:
        names = folders
    if kl_threshold is None:
        kl_threshold = [0.05 for _ in range(len(folders))]
    if env_num is None:
        env_num = [32 for _ in folders]
    if batch_size is None:
        batch_size = [8 * 16 for _ in folders]
    plt.figure()
    # ES_record = {}
    all_folders_ES_idx = {}
    for n, folder in enumerate(folders):
        all_seeds_stat = []
        all_seeds_ES_idx = []
        for seed in os.listdir(folder):
            # if seed == '1409' and folder == 'ft_baseline_KL_5_wo_PE+dropout':
            #     continue
            if 'checkpoint_100.pt' not in os.listdir(os.path.join(folder, seed)):
                continue
            with open(os.path.join(folder, seed, 'Unimal-v0_results.json'), 'r') as f:
                log = json.load(f)
            try:
                data = np.array(log['__env__'][key])
            except:
                continue
            approx_kl = np.array(log['__env__']['approx_kl'])
            
            iter_avg = []
            early_stop_idx = []
            i = 0
            while i < len(data):
                # check for early stop
                early_stop_pos = np.where(approx_kl[i:(i + batch_size[n])] > kl_threshold[n])[0]
                if len(early_stop_pos) > 0:
                    end_index = i + early_stop_pos[0] + 1
                    early_stop_idx.append(early_stop_pos[0])
                else:
                    end_index = i + batch_size[n]
                    early_stop_idx.append(batch_size[n])

                avg = data[i:end_index].mean()
                iter_avg.append(avg)
                i = end_index

            all_seeds_stat.append(iter_avg)
            # plt.plot(iter_avg, label=f'{folder}-{seed}')
            all_seeds_ES_idx.append(np.array(early_stop_idx))
        if len(all_seeds_stat) == 0:
            continue
        min_len = min([len(x) for x in all_seeds_stat])
        avg_stat = np.stack([np.array(x[:min_len]) for x in all_seeds_stat]).mean(0)
        all_folders_ES_idx[folder] = np.stack([x[:min_len] for x in all_seeds_ES_idx]).mean(0)
        plt.plot([2560 * env_num[n] * i for i in range(len(avg_stat))], avg_stat, label=names[n])
    plt.legend()
    plt.xlabel('update num')
    plt.ylabel(key)
    if prefix is None:
        plt.savefig(f'figures/train_stats/{key}.png')
    else:
        plt.savefig(f'figures/train_stats/{prefix}_{key}.png')
    plt.close()

    plt.figure()
    print (all_folders_ES_idx.keys())
    w = 10
    for i, folder in enumerate(folders):
        y_data = np.convolve(all_folders_ES_idx[folder], np.ones(w), 'valid') / w
        plt.plot([2560 * env_num[i] * j for j in range(len(y_data))], y_data, \
            label=f'{names[i]}: {all_folders_ES_idx[folder].mean():.2f}')
    plt.legend()
    plt.savefig(f'figures/train_stats/{prefix}_ES_index.png')
    plt.close()


def plot_test_performance(results):
    scores = []
    for r in results:
        with open(f'eval/{r}.pkl', 'rb') as f:
            score = pickle.load(f)
            scores.append(score)
    
    idx = np.argsort(-scores[0])
    plt.figure()
    for i, score in enumerate(scores):
        plt.plot(score[idx], label=results[i])
        print (results[i], score.mean())
    plt.legend()
    plt.savefig('figures/test_results.png')
    plt.close()


def scatter_train_performance_compared_to_subset_train(folder, option):

    MT_agent_score = defaultdict(list)
    for seed in os.listdir('output/' + folder):
        with open(os.path.join('output', folder, seed, 'Unimal-v0_results.json'), 'r') as f:
            log = json.load(f)
        del log['__env__']
        del log['fps']
        for agent in log:
            s = log[agent]['reward']['reward'][-1]
            MT_agent_score[agent].append(s)

    subset_agent_score = defaultdict(list)
    for index in [0, 1, 2]:
        folder = f'output/log_train_subset_{index}'
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        del log['__env__']
        del log['fps']
        for agent in log:
            s = log[agent]['reward']['reward'][-1]
            subset_agent_score[agent].append(s)

    ST_agent_score = defaultdict(list)
    ST_folder = 'output/log_single_task'
    for agent in os.listdir(ST_folder):
        for seed in os.listdir(os.path.join(ST_folder, agent)):
            path = os.path.join(ST_folder, agent, seed)
            # use this if check to avoid the seed which is not fully trained
            if 'checkpoint_100.pt' in os.listdir(path):
                with open(f'{path}/Unimal-v0_results.json', 'r') as f:
                    log = json.load(f)
                s = log[agent]['reward']['reward'][-1]
                ST_agent_score[agent].append(s)

    if option == 'ST-subset':
        x = ST_agent_score
        y = subset_agent_score
        x_label = 'single task train'
        y_label = 'train on 10 morphologies'
        fig_name = 'compare_final_train_score_ST_subset'

    plt.figure()
    for agent in subset_agent_score:
        plt.scatter(np.array(x[agent]).mean(), np.array(y[agent]).mean(), c='blue')
    plt.plot([500, 5500], [500, 5500], 'k-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'figures/{fig_name}.png')
    plt.close()


def get_context():

    cfg.ENV.WALKER_DIR = 'unimals_100/test'
    cfg.ENV.WALKERS = []
    cfg.OUT_DIR = './eval'
    cfg.PPO.NUM_ENVS = 2
    set_cfg_options()

    agents = list(os.listdir('unimals_single_task_test'))
    context = {}
    for agent in agents:
        env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        obs = env.reset()
        # context[agent] = obs['context'].reshape(cfg.MODEL.MAX_LIMBS, -1)[:env.metadata["num_limbs"]]
        env.close()
    
    # with open('train_context.pkl', 'wb') as f:
        # pickle.dump(context, f)


def analyze_ratio_hist(folders):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            for epoch in range(8):
                if len(hist[epoch]) == 0:
                    break
                batch_avg = np.stack([np.array(x) for x in hist[epoch]]).mean(axis=0)
                plt.subplot(2, 4, epoch + 1)
                plt.plot(np.linspace(0., 2., 100), batch_avg, label=folder, c=colors[j])
                plt.plot([1., 1.], [0, batch_avg.max()], '-k')
        plt.legend()
        plt.savefig(f'figures/ratio_hist/{i}.png')
        plt.close()


def analyze_ratio_hist_trend(folders, prefix=None):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            plt.subplot(1, 2, j + 1)
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            for epoch in range(8):
                if len(hist[epoch]) == 0:
                    break
                batch_avg = np.stack([np.array(x) for x in hist[epoch]]).mean(axis=0)
                plt.plot(np.linspace(0., 2., 100), batch_avg, c=colors[j], alpha=0.3 + epoch * 0.1)
                plt.plot([1., 1.], [0, batch_avg.max()], '--k')
            plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/trend_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/trend_{prefix}_{i}.png')
        plt.close()


def test_init_ratio_hist(folders, prefix=None):
    seed = 1410
    colors = ['blue', 'orange']

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        for j, folder in enumerate(folders):
            plt.subplot(1, 2, j + 1)
            if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
                return
            with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
                hist = pickle.load(f)
            batch_init = np.array(hist[0][0])
            plt.plot(np.linspace(0., 2., 100), batch_init, c=colors[j])
            plt.plot([1., 1.], [0, batch_init.max()], '-k')
            plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/init_ratio_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/init_ratio_{prefix}_{i}.png')
        plt.close()


def scatter_test_score(x, y, name_x, name_y):
    plt.figure()
    for agent in x:
        plt.scatter(np.array(x[agent]).mean(), np.array(y[agent]).mean())
    plt.plot([500, 5500], [500, 5500], 'k-')
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.savefig(f'figures/compare_test_score/{name_x}-{name_y}.png')
    plt.close()


def plot_test_score(files, fig_name):
    scores = {}
    for i, x in enumerate(files):
        with open(f'eval/{x}.pkl', 'rb') as f:
            score = pickle.load(f)
        scores[x] = score
        if i == 0:
            agent_list = list(score.keys())
            avg_score = np.array([score[agent].mean() for agent in agent_list])
            order = np.argsort(avg_score)
    
    plt.figure()
    for f in scores:
        all_agent_score = np.stack([scores[f][agent] for agent in agent_list])
        all_agent_score = all_agent_score[order]
        avg_score = all_agent_score.mean(axis=1)
        std_score = all_agent_score.std(axis=1)
        plt.plot(avg_score, label=f'{f}: {avg_score.mean():.0f} +- {std_score.mean():.0f}')
        plt.fill_between(np.arange(len(avg_score)), avg_score - std_score, avg_score + std_score, alpha=0.25)
    plt.legend()
    plt.savefig(f'figures/compare_test_score/{fig_name}.png')
    plt.close()


def scatter_test_score(x, y):
    suffix = [
        '_terminate_on_fall_deterministic', 
        '_terminate_on_fall', 
        '_deterministic', 
        '', 
    ]

    plt.figure(figsize=(16, 12))
    for i, suf in enumerate(suffix):
        with open(f'eval/{x}{suf}.pkl', 'rb') as f:
            result_x = pickle.load(f)
        with open(f'eval/{y}{suf}.pkl', 'rb') as f:
            result_y = pickle.load(f)
        agents = list(result_x.keys())
        avg_x = np.array([np.array(result_x[r]).mean() for r in agents])
        avg_y = np.array([np.array(result_y[r]).mean() for r in agents])
        std_x = np.array([np.array(result_x[r]).std() for r in agents])
        std_y = np.array([np.array(result_y[r]).std() for r in agents])
        plt.subplot(2, 2, i + 1)
        plt.plot([0, 6000], [0, 6000], '--k')
        plt.scatter(avg_x, avg_y)
        if i == 0:
            plt.xlabel(f'{x}: {avg_x.mean():.2f} +- {std_x.mean():.2f}')
            plt.ylabel(f'{y}: {avg_y.mean():.2f} +- {std_y.mean():.2f}')
        else:
            plt.xlabel(f'{avg_x.mean():.2f} +- {std_x.mean():.2f}')
            plt.ylabel(f'{avg_y.mean():.2f} +- {std_y.mean():.2f}')
        plt.title(suf)
    plt.savefig(f'figures/compare_test_score/compare_score_{x}-vs-{y}.png')
    plt.close()


def scatter_train_score(x, y):

    score_x = {}
    with open(f'output/{x}/Unimal-v0_results.json', 'r') as f:
        log = json.load(f)
    del log['__env__']
    del log['fps']
    for agent in log:
        score_x[agent] = log[agent]['reward']['reward'][-1]

    score_y = {}
    with open(f'output/{y}/Unimal-v0_results.json', 'r') as f:
        log = json.load(f)
    del log['__env__']
    del log['fps']
    for agent in log:
        score_y[agent] = log[agent]['reward']['reward'][-1]
    
    agents = list(score_x.keys())
    plt.figure()
    plt.plot([500, 5500], [500, 5500], '--k')
    plt.scatter([score_x[agent] for agent in agents], [score_y[agent] for agent in agents])
    plt.xlabel(x)
    plt.ylabel(y)
    x = x.replace('/', '_')
    y = y.replace('/', '_')
    plt.savefig(f'figures/compare_train_score/{x}-vs-{y}.png')
    plt.close()


def analyze_ratio_hist_per_epoch(folder, prefix=None, seed=1409):

    for i in range(0, 1200, 50):
        plt.figure(figsize=(16, 9))
        if f'ratio_hist_{i}.pkl' not in os.listdir(f'output/{folder}/{seed}/ratio_hist'):
            return
        with open(f'output/{folder}/{seed}/ratio_hist/ratio_hist_{i}.pkl', 'rb') as f:
            hist = pickle.load(f)
        for epoch in range(8):
            plt.subplot(2, 4, epoch + 1)
            if len(hist[epoch]) == 0:
                break
            batch_ratio = hist[epoch]
            for j, batch in enumerate(batch_ratio):
                plt.plot(np.linspace(0., 2., 100), batch, c='b', alpha=0.25 + epoch * 0.05)
                break
            plt.plot([1., 1.], [0, 500], '--k')
        plt.legend()
        if prefix is None:
            plt.savefig(f'figures/ratio_hist/epoch_viz_{i}.png')
        else:
            plt.savefig(f'figures/ratio_hist/epoch_viz_{prefix}_{i}.png')
        plt.close()


def analyze_modular_obs_scale(morphology, seed=1409):
    model_type = [
        # 'TF', 
        # 'TF_limb_norm', 
        'TF_wo_vecnorm', 
    ]
    all_agent_min, all_agent_max = np.zeros(19), np.zeros(19)
    for n, agent in enumerate(os.listdir(f'modular/{morphology}s/xml')):
        agent_name = agent.split('.')[0]
        print (agent_name)
        obs_all = {}
        for model in model_type:
            record_file = f'obs_modular_{morphology}_train_{model}_1409_{morphology}s_{agent_name}.pkl'
            with open(f'eval_history/{record_file}', 'rb') as f:
                obs = pickle.load(f)
            obs_all[model] = obs.reshape(-1, 19)
        
        plt.figure()
        for i in range(19):
            plt.figure()
            for j, model in enumerate(model_type):
                plt.subplot(1, 3, j + 1)
                plt.hist(obs_all[model][:, i], bins=100)
                plt.title(model)
                if n == 0:
                    all_agent_min[i] = obs_all[model][:, i].min()
                    all_agent_max[i] = obs_all[model][:, i].max()
                else:
                    all_agent_min[i] = min(all_agent_min[i], obs_all[model][:, i].min())
                    all_agent_max[i] = max(all_agent_max[i], obs_all[model][:, i].max())
            plt.savefig(f'figures/modular_obs/{agent}_{i}.png')
            plt.close()
    for i in range(19):
        print (i, all_agent_min[i], all_agent_max[i])


def analyze_modular_embedding_PE_scale(morphology, seed=1409):
    model_type = [
        'TF', 
        'TF_wo_vecnorm', 
        'TF_wo_vecnorm_vanilla_PE', 
        'TF_wo_vecnorm_semantic_PE', 
    ]
    for agent in os.listdir(f'modular/{morphology}s/xml'):
        agent_name = agent.split('.')[0]
        embedding_norm = {}
        for model in model_type:
            record_file = f'embedding_modular_{morphology}_train_{model}_1409_{morphology}s_{agent_name}.pkl'
            with open(f'eval_history/{record_file}', 'rb') as f:
                embeddings = pickle.load(f)
            norm = np.linalg.norm(embeddings, ord=2, axis=-1).ravel()
            embedding_norm[model] = norm
        
        plt.figure()
        for j, model in enumerate(model_type):
            plt.subplot(2, 2, j + 1)
            plt.hist(embedding_norm[model], bins=100)
            plt.title(model)
        plt.savefig(f'figures/modular_embedding/{agent}.png')
        plt.close()


def analyze_train_robot_num():
    folders = [
        'ft_baseline_subset_10_KL_5_wo_PE+dropout', 
        'ft_baseline_subset_20_KL_5_wo_PE+dropout', 
        'ft_baseline_subset_50_KL_5_wo_PE+dropout', 
        'ft_baseline_KL_5_wo_PE+dropout', 
        'ft_baseline_train+joint_angle_KL_5_wo_PE+dropout', 
        'ft_baseline_train+limb_params_KL_5_wo_PE+dropout', 
        'ft_baseline_train+limb_params+joint_angle_KL_5_wo_PE+dropout', 
        'ft_random_joint_angle_KL_5_wo_PE+dropout', 
    ]
    names = [
        '10', 
        '20', 
        '50', 
        '100', 
        '100 + 400 joint angle', 
        '100 + 400 limb params', 
        '100 + 800 both', 
        '100 + random joint angle'
    ]

    all_methods_scores = []
    for folder in folders:
        print (folder)
        scores = []
        for seed in [1409, 1410]:
            with open(f'eval/{folder}_{seed}_test.pkl', 'rb') as f:
                results = pickle.load(f)
            avg_score = np.array([np.array(results[agent][0]).mean() for agent in results]).mean()
            print (seed, avg_score)
            scores.append(avg_score)
        all_methods_scores.append(np.array(scores))
    
    plt.figure()
    plt.bar(np.arange(len(all_methods_scores)), [s.mean() for s in all_methods_scores])
    plt.errorbar(np.arange(len(all_methods_scores)), [s.mean() for s in all_methods_scores], yerr=[s.std() for s in all_methods_scores], fmt='none', c='k')
    plt.xticks(np.arange(len(all_methods_scores)), names, rotation=30, ha='right')
    plt.xlabel('training robots')
    plt.ylabel('test performance')
    plt.tight_layout()
    plt.savefig('figures/train_robot_num.png')
    plt.close()


def compare_evaluation_results():
    folders = {
        'ft_baseline_KL_5_wo_PE+dropout': ['train', 'test', 'joint_angle'], 
        'ft_baseline_train+test_KL_5_wo_PE+dropout': ['train', 'test'], 
        'ft_random_joint_angle_KL_5_wo_PE+dropout': ['train', 'test', 'joint_angle'], 
        'ft_baseline_train+joint_angle_KL_5_wo_PE+dropout': ['train', 'test', 'joint_angle'], 
    }

    folders = {
        'ft_baseline_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_baseline_train+joint_angle_300_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_random_joint_angle_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_baseline_train+limb_params_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
    }

    folders = {
        'ft_baseline_train+joint_angle_300_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_filter_2000_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_filter_3000_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_filter_4000_KL_5_wo_PE+dropout': ['train', 'joint_angle_100', 'test'], 
    }

    folders = {
        'ft_baseline_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_baseline_train+joint_angle_300_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_uniform_task_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_UED_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_train+joint_angle_300_HN+FA_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_random_joint_angle_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_400M_train_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_400M_train+joint_angle_300_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
        'ft_400M_train+joint_angle_random_KL_5_wo_PE+dropout': ['joint_angle_100', 'test'], 
    }

    # folders = {
    #     'ft_baseline_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_train_mutate_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_train_mutate_HN+FA_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': ['test'], 
    # }

    # folders = {
    #     'ft_random_100_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_random_400_KL_5_wo_PE+dropout': ['test'], 
    #     'ft_random_1000_KL_5_wo_PE+dropout': ['test'], 
    # }

    for folder in folders:
        print (folder)
        for test_set in folders[folder]:
            scores = []
            for seed in [1409]:
                if not os.path.exists(f'eval/{folder}_{seed}_{test_set}.pkl'):
                    continue
                with open(f'eval/{folder}_{seed}_{test_set}.pkl', 'rb') as f:
                    results = pickle.load(f)
                avg_score = np.array([np.array(results[agent][0]).mean() for agent in results]).mean()
                scores.append(avg_score)
            print (test_set, np.array(scores).mean(), np.array(scores).std(), f'{len(scores)} seeds')


def source_policy_logp(source_folder, agent_path):

    from metamorph.config import cfg
    cfg.merge_from_file(f'{source_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'{source_folder}/Unimal-v0.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    set_cfg_options()
    source_trainer = PPO()
    cfg.PPO.NUM_ENVS = 32
    source_policy = source_trainer.agent
    source_policy.ac.eval()

    cfg.MODEL.OBS_TO_NORM = []
    env = make_vec_envs(training=False, norm_rew=False)
    # set_ob_rms(env, get_ob_rms(source_trainer.envs))
    obs_rms = get_ob_rms(source_trainer.envs)['proprioceptive']
    mean = torch.from_numpy(obs_rms.mean).to(torch.float32).cuda()
    std = torch.sqrt(torch.from_numpy(obs_rms.var + 1e-8)).to(torch.float32).cuda()
    
    obs = env.reset()
    trajectory = []
    for t in range(2000):
        obs_norm = copy.deepcopy(obs)
        obs_norm['proprioceptive'] = torch.clip((obs['proprioceptive'] - mean) / std, -10., 10).to(torch.float32)
        _, act, logp, _, _ = source_policy.act(obs_norm, return_attention=False, compute_val=False)
        trajectory.append([obs, act, logp])
        obs, reward, done, infos = env.step(act)
    env.close()
    return trajectory


def target_policy_logp(target_folder, trajectory):
    # load the target agent
    from metamorph.config import cfg
    cfg.merge_from_file(f'{target_folder}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'{target_folder}/Unimal-v0.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    set_cfg_options()
    target_trainer = PPO()
    target_policy = target_trainer.agent
    target_policy.ac.eval()

    obs_rms = get_ob_rms(target_trainer.envs)['proprioceptive']
    mean = torch.from_numpy(obs_rms.mean).to(torch.float32).cuda()
    std = torch.sqrt(torch.from_numpy(obs_rms.var + 1e-8)).to(torch.float32).cuda()

    logp_record = []
    for t in range(2000):
        obs, act = trajectory[t][0], trajectory[t][1]
        obs_norm = copy.deepcopy(obs)
        obs_norm['proprioceptive'] = torch.clip((obs['proprioceptive'] - mean) / std, -10., 10).to(torch.float32)
        val, pi, logp, _, _, _ = target_policy.ac(obs_norm, act=act, return_attention=False, compute_val=False)
        logp_record.append(logp)
    return logp_record


def compare_policy_divergence(source_folder, target_folder, agent_path):
    
    trajectory = source_policy_logp(source_folder, agent_path)
    source_logp = torch.stack([x[-1] for x in trajectory])

    target_logp = target_policy_logp(target_folder, trajectory)
    target_logp = torch.stack(target_logp)

    divergence = (source_logp - target_logp).mean().item()
    return divergence


def plot_ST_score(folders, label):
    scores = {}
    for folder in folders:
        scores[folder] = []
        for agent in os.listdir(folder):
            with open(f'{folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
                log = json.load(f)
            scores[folder].append(log[agent]['reward']['reward'][-1])
    
    plt.figure()
    for folder in folders:
        plt.plot(np.sort(np.array(scores[folder])), label=folder)
    plt.legend()
    plt.title('sorted single-robot training performance')
    plt.savefig(f'figures/ST_task_return_{label}.png')
    plt.close()


def generalization_curve(folders, names, test_set, suffix, height_check=False, plot_each_seed=False, seeds=[1409, 1410]):

    colors = plt.get_cmap('tab10').colors

    plt.figure()
    for i, folder in enumerate(folders):
        print (folder)
        try:
            cfg.merge_from_file(f'output/{folder}/1409/config.yaml')
        except:
            cfg.merge_from_file(f'baselines/{folder}/1409/config.yaml')
        all_seed_x, all_seed_y = [], []
        for seed in seeds:
            xdata, ydata = [0], [0.]
            iteration = folders[folder]
            while (1):
                file_path = f'eval/{folder}/{seed}_{test_set}_cp_{iteration}.pkl'
                if not os.path.exists(file_path):
                    break
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                avg_score = np.array([np.array(results[agent][0]).mean() for agent in results]).mean()
                print (iteration, len(results.keys()))
                xdata.append(iteration)
                ydata.append(avg_score)
                iteration += folders[folder]
            if iteration != folders[folder]:
                all_seed_x.append(xdata)
                all_seed_y.append(ydata)
        seed_num = len(all_seed_x)
        print (all_seed_y)
        min_len = min([len(x) for x in all_seed_x])
        plot_x = all_seed_x[0][:min_len]
        plot_x = np.array(plot_x) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS
        plot_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).mean(axis=0)
        error_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).std(axis=0)
        # print (plot_x, plot_y)
        print (f'final score: {plot_y[-1]} +- {error_y[-1]}')
        plt.errorbar(plot_x, plot_y, yerr=error_y, c=colors[i], label=f'{names[folder]} ({seed_num} seeds)')
        if plot_each_seed:
            for y in all_seed_y:
                plt.plot(plot_x, y[:min_len], c=colors[i], alpha=0.5)
        
        if height_check:
            all_seed_x, all_seed_y = [], []
            for seed in seeds:
                xdata, ydata = [0], [0.]
                iteration = folders[folder]
                while (1):
                    file_path = f'eval/{folder}/{test_set}_cp_{iteration}_wo_height_check.pkl'
                    if not os.path.exists(file_path):
                        break
                    with open(file_path, 'rb') as f:
                        results = pickle.load(f)
                    avg_score = np.array([np.array(results[agent][0]).mean() for agent in results]).mean()
                    print (iteration, len(results.keys()))
                    xdata.append(iteration)
                    ydata.append(avg_score)
                    iteration += folders[folder]
                if iteration != folders[folder]:
                    all_seed_x.append(xdata)
                    all_seed_y.append(ydata)
            seed_num = len(all_seed_x)
            print (all_seed_y)
            min_len = min([len(x) for x in all_seed_x])
            plot_x = all_seed_x[0][:min_len]
            plot_x = np.array(plot_x) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS
            plot_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).mean(axis=0)
            error_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).std(axis=0)
            # print (plot_x, plot_y)
            print (f'final score: {plot_y[-1]} +- {error_y[-1]}')
            plt.errorbar(plot_x, plot_y, yerr=error_y, c=colors[i + len(folders)], label=f'{names[folder]} (wo height check) ({seed_num} seeds)')
            if plot_each_seed:
                for y in all_seed_y:
                    plt.plot(plot_x, y[:min_len], c=colors[i], alpha=0.5)
    plt.legend()
    plt.title(f'Generalization to {test_set}')
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    plt.savefig(f'figures/generalization_curve_{test_set}_{suffix}.png')
    plt.close()


def per_agent_generalization_curve(folders, names, test_set, suffix):

    colors = plt.get_cmap('tab10').colors

    agents = [x[:-4] for x in os.listdir(f'unimals_100/{test_set}/xml')]

    data = {}
    for i, folder in enumerate(folders):
        print (folder)
        cfg.merge_from_file(f'output/{folder}/1409/config.yaml')
        all_seed_x, all_seed_y = [], []
        for seed in [1409, 1410]:
            xdata, ydata = [0], [[0. for _ in agents]]
            iteration = folders[folder]
            while (1):
                file_path = f'eval/{folder}/{folder}_{seed}_{test_set}_cp_{iteration}.pkl'
                if not os.path.exists(file_path):
                    break
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                per_agent_avg_score = [np.array(results[agent][0]).mean() for agent in agents]
                xdata.append(iteration)
                ydata.append(per_agent_avg_score)
                iteration += folders[folder]
            # each column corresponds to the generalization curve of one agent
            ydata = np.stack(ydata, axis=0)
            if iteration != folders[folder]:
                all_seed_x.append(xdata)
                all_seed_y.append(ydata)
        seed_num = len(all_seed_x)
        min_len = min([len(x) for x in all_seed_x])
        xdata = all_seed_x[0][:min_len]
        ydata = np.stack([y[:min_len] for y in all_seed_y])
        data[folder] = [xdata, ydata]

    plt.figure(figsize=(20, 20))
    for i, agent in enumerate(agents):
        if i >= 100:
            break
        plt.subplot(10, 10, i + 1)
        for j, folder in enumerate(folders):
            plot_x = np.array(data[folder][0]) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS
            plot_y = data[folder][1][:, :, i].mean(axis=0)
            error_y = data[folder][1][:, :, i].std(axis=0)
            plt.errorbar(plot_x, plot_y, yerr=error_y, c=colors[j], label=names[folder])
            plt.ylim(-100, 5500)
    plt.legend()
    plt.savefig(f'figures/per_agent_generalization_curve_{test_set}_{suffix}.png')
    plt.savefig(f'figures/per_agent_generalization_curve_{test_set}_{suffix}.pdf')
    plt.close()


# plot the std across different evaluation runs on a single seed
def per_agent_generalization_curve_v2(folders, names, test_set, suffix):

    colors = plt.get_cmap('tab10').colors

    agents = [x[:-4] for x in os.listdir(f'unimals_100/{test_set}/xml')]

    data = {}
    for i, folder in enumerate(folders):
        print (folder)
        cfg.merge_from_file(f'output/{folder}/1409/config.yaml')
        xdata, ydata, yerror = [0], [[0. for _ in agents]], [[0. for _ in agents]]
        iteration = folders[folder]
        while (1):
            file_path = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}.pkl'
            if not os.path.exists(file_path):
                break
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            per_agent_avg, per_agent_std = [], []
            for agent in agents:
                if agent in results:
                    per_agent_avg.append(np.array(results[agent][0]).mean())
                    per_agent_std.append(np.array(results[agent][0]).std())
                else:
                    per_agent_avg.append(0.)
                    per_agent_std.append(0.)
            # per_agent_avg = [np.array(results[agent][0]).mean() for agent in agents]
            # per_agent_std = [np.array(results[agent][0]).std() for agent in agents]
            xdata.append(iteration * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS)
            ydata.append(per_agent_avg)
            yerror.append(per_agent_std)
            iteration += folders[folder]
        # each column corresponds to the generalization curve of one agent
        ydata = np.stack(ydata, axis=0)
        yerror = np.stack(yerror, axis=0)
        data[folder] = [xdata, ydata, yerror]

    row_num = math.ceil(ydata.shape[1] / 10)
    plt.figure(figsize=(20, 2 * row_num))
    for i, agent in enumerate(agents):
        plt.subplot(row_num, 10, i + 1)
        for j, folder in enumerate(folders):
            plot_x = np.array(data[folder][0])
            plot_y = data[folder][1][:, i]
            error_y = data[folder][2][:, i]
            plt.errorbar(plot_x, plot_y, yerr=error_y, c=colors[j], label=names[folder])
            plt.ylim(-100, 5500)
    plt.legend()
    # plt.tight_layout()
    plt.savefig(f'figures/per_agent_generalization_curve_{test_set}_{suffix}.png')
    plt.savefig(f'figures/per_agent_generalization_curve_{test_set}_{suffix}.pdf')
    plt.close()


def compare_MT_ST(ST_folder, MT_eval_file, suffix):

    agents = os.listdir(f'output/{ST_folder}')

    ST_results = {}
    for agent in agents:
        ST_results[agent] = 0.
        seed_count = 0
        for seed in os.listdir(f'output/{ST_folder}/{agent}'):
            if 'checkpoint_-1.pt' not in os.listdir(f'output/{ST_folder}/{agent}/{seed}'):
                continue
            with open(f'output/{ST_folder}/{agent}/{seed}/Unimal-v0_results.json', 'r') as f:
                log = json.load(f)
            ST_results[agent] += np.array(log[agent]['reward']['reward'][-10:]).mean()
            seed_count += 1
        if seed_count == 0:
            del ST_results[agent]
        else:
            ST_results[agent] /= seed_count

    agents = list(ST_results.keys())
    ST_results = [ST_results[agent] for agent in agents]

    with open(f'eval/{MT_eval_file}.pkl', 'rb') as f:
        results = pickle.load(f)
    MT_results = []
    for agent in agents:
        MT_results.append(results[agent][0].mean())
    
    plt.figure()
    plt.scatter(ST_results, MT_results)
    lower_bound = 0.95 * min([min(ST_results), min(MT_results)])
    upper_bound = 1.05 * max([max(ST_results), max(MT_results)])
    plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'k')
    plt.xlabel('Single-robot training')
    plt.ylabel('Multi-robot training')
    plt.legend()
    plt.savefig(f'figures/MT_ST_{suffix}.png')
    plt.close()


def compare_MT_ST_v2(ST_folder, MT_folder, suffix):

    agents = os.listdir(f'output/{ST_folder}')

    ST_results = {}
    for agent in agents:
        ST_results[agent] = 0.
        seed_count = 0
        for seed in os.listdir(f'output/{ST_folder}/{agent}'):
            if 'checkpoint_-1.pt' not in os.listdir(f'output/{ST_folder}/{agent}/{seed}'):
                continue
            with open(f'output/{ST_folder}/{agent}/{seed}/Unimal-v0_results.json', 'r') as f:
                log = json.load(f)
            ST_results[agent] += np.array(log[agent]['reward']['reward'][-10:]).mean()
            seed_count += 1
        if seed_count == 0:
            del ST_results[agent]
        else:
            ST_results[agent] /= seed_count

    agents = list(ST_results.keys())
    ST_results = [ST_results[agent] for agent in agents]

    with open(f'output/{MT_folder}/Unimal-v0_results.json', 'r') as f:
        log = json.load(f)
    MT_results = {}
    for agent in agents:
        MT_results[agent] = np.mean(log[agent]['reward']['reward'][-10:])
    MT_results = [MT_results[agent] for agent in agents]
    
    plt.figure()
    plt.scatter(ST_results, MT_results)
    lower_bound = 0.95 * min([min(ST_results), min(MT_results)])
    upper_bound = 1.05 * max([max(ST_results), max(MT_results)])
    plt.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'k')
    plt.xlabel('Single-robot training')
    plt.ylabel('Multi-robot training')
    plt.legend()
    plt.savefig(f'figures/MT_ST_{suffix}.png')
    plt.close()


def move_eval():
    for eval_result in os.listdir('eval/'):
        if eval_result[-3:] != 'pkl':
            continue
        seed = None
        if '1409' in eval_result:
            seed = '1409'
        elif '1410' in eval_result:
            seed = '1410'
        elif '1411' in eval_result:
            seed = '1411'
        if not seed:
            continue
        folder_name = eval_result.split(f'_{seed}_')[0]
        os.makedirs(f'eval/{folder_name}', exist_ok=True)
        os.system(f'mv eval/{eval_result} eval/{folder_name}/')


# check the sampling frequency of each agent in the last training iteration
def check_env_sample_freq(folder, ST_score):
    # plt.figure()
    # for seed in os.listdir(folder):
    #     with open(f'{folder}/{seed}/sampling.json', 'r') as f:
    #         samples = json.load(f)
    #     count = np.bincount(np.array(samples), minlength=400)
    #     freq = count / count.sum()
    #     plt.plot(np.sort(freq), label=seed)
    # plt.legend()
    # folder_name = folder.split('/')[1]
    # plt.title(folder_name)
    # plt.savefig(f'figures/env_sample_distribution_{folder_name}.png')
    # plt.close()

    plt.figure()
    for i, seed in enumerate(os.listdir(folder)):
        with open(f'{folder}/{seed}/sampling.json', 'r') as f:
            samples = json.load(f)
        count = np.bincount(np.array(samples), minlength=400)
        freq = count / count.sum()
        cfg.merge_from_file(f'{folder}/{seed}/config.yaml')
        agents = cfg.ENV.WALKERS
        ST_final_score = [ST_score[agent] for agent in agents]
        plt.subplot(1, 2, i + 1)
        plt.scatter(freq, ST_final_score)
    folder_name = folder.split('/')[1]
    plt.savefig(f'figures/check_{folder_name}.png')
    plt.close()


# check how each agent's score and sampling frequency change under different UED strategies
def check_UED_trend(folders, max_iter=1219):

    all_probs, all_freqs, all_scores = {}, {}, {}
    for folder in folders:
        # load agents
        cfg.merge_from_file(f'{folder}/config.yaml')
        agents = cfg.ENV.WALKERS

        # get the sampling prob
        probs = defaultdict(list)
        for i in range(max_iter):
            with open(f'{folder}/iter_prob/{i-1}.pkl', 'rb') as f:
                prob = pickle.load(f)
            for j, agent in enumerate(agents):
                probs[agent].append(prob[j])
        all_probs[folder] = probs
        
        # get the actually sampled freq
        freqs = defaultdict(list)
        for i in range(max_iter):
            with open(f'{folder}/iter_sampled_agents/{i}.pkl', 'rb') as f:
                sampled_agents = pickle.load(f)
            sample_num = len(sampled_agents)
            for agent in agents:
                freq = sampled_agents.count(agent) / sample_num
                freqs[agent].append(freq)
        all_freqs[folder] = freqs
        
        # get the agent's average training score after each iteration
        return_curve = defaultdict(list)
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            results = json.load(f)
        for agent in agents:
            return_curve[agent] = results[agent]['reward']['reward']
        all_scores[folder] = return_curve

    n_folder = len(folders)
    for agent in agents:
        plt.figure(figsize=(16, 8))
        for i, y in enumerate([all_probs, all_scores]):
            plt.subplot(1, 2, i + 1)
            for folder in folders:
                plt.plot(y[folder][agent], label=folder)
        plt.legend()
        plt.savefig(f'figures/check_UED/{agent}.png')
        plt.close()


def check_ACCEL(folders, max_iter=1219):

    def normalize_staleness_score(staleness_score, cur_iter):
        normalized_staleness_score = cur_iter - staleness_score + 1.
        normalized_staleness_score /= normalized_staleness_score.sum()
        return normalized_staleness_score
    
    def normalize_potential_score(potential_score):
        order = np.argsort(-potential_score)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order)) + 1
        normalized_potential_score = 1. / rank
        normalized_potential_score /= (normalized_potential_score).sum()
        return normalized_potential_score

    all_potential_scores, all_staleness_scores, all_probs, all_scores = {}, {}, {}, {}
    for folder in folders:
        # load agents
        cfg.merge_from_file(f'{folder}/config.yaml')
        agents = cfg.ENV.WALKERS

        # get ACCEL scores
        try:
            potential_score, staleness_score = defaultdict(list), defaultdict(list)
            for i in range(10, max_iter):
                with open(f'{folder}/ACCEL_score/{i}.pkl', 'rb') as f:
                    p, s = pickle.load(f)
                p = normalize_potential_score(np.array(p))
                s = normalize_staleness_score(np.array(s), i)
                for j, agent in enumerate(agents):
                    potential_score[agent].append(p[j])
                    staleness_score[agent].append(s[j])
            all_potential_scores[folder] = potential_score
            all_staleness_scores[folder] = staleness_score
        except:
            pass

        # get the sampling prob
        probs = defaultdict(list)
        for i in range(max_iter):
            try:
                with open(f'{folder}/iter_prob/{i-1}.pkl', 'rb') as f:
                    prob = pickle.load(f)
                for j, agent in enumerate(agents):
                    probs[agent].append(prob[j])
            except:
                break
        all_probs[folder] = probs
        
        # get the agent's average training score after each iteration
        return_curve = defaultdict(list)
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            results = json.load(f)
        for agent in agents:
            return_curve[agent] = results[agent]['reward']['reward']
        all_scores[folder] = return_curve

    n_folder = len(folders)
    for agent in agents:
        plt.figure(figsize=(16, 16))
        for i, y in enumerate([all_potential_scores, all_staleness_scores, all_probs, all_scores]):
            plt.subplot(2, 2, i + 1)
            for folder in folders:
                try:
                    if i in [0, 1]:
                        plt.plot(range(10, max_iter), y[folder][agent], label=folder)
                    else:
                        plt.plot(y[folder][agent], label=folder)
                except:
                    continue
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figures/check_ACCEL/{agent}.png')
        plt.close()


def analyze_L1_value_loss(folder, name, max_iter=1219):

    cfg.merge_from_file(f'{folder}/config.yaml')
    agents = cfg.ENV.WALKERS

    # get the sampling prob
    probs = defaultdict(list)
    for i in range(max_iter):
        try:
            with open(f'{folder}/iter_prob/{i-1}.pkl', 'rb') as f:
                prob = pickle.load(f)
            for j, agent in enumerate(agents):
                probs[agent].append(prob[j])
        except:
            break
    
    # order the agents by the total sampling probs
    total_probs = [sum(probs[agent]) for agent in agents]
    order = np.argsort(-np.array(total_probs))
    
    # merge into a matrix and cumsum
    all_probs = np.stack([probs[agents[i]] for i in order])
    cumsum = np.cumsum(all_probs, axis=0)

    cmap = plt.cm.get_cmap('viridis')
    plt.figure()
    for i in range(cumsum.shape[0]):
        if i == 0:
            plt.fill_between(np.arange(cumsum.shape[1]), cumsum[i], color=cmap(i/cumsum.shape[0]))
        else:
            plt.fill_between(np.arange(cumsum.shape[1]), cumsum[i], cumsum[i-1], color=cmap(i/cumsum.shape[0]))
    plt.plot(cumsum[9], 'k--', label='top 10 agents')
    plt.plot(cumsum[19], 'r--', label='top 20 agents')
    plt.plot(cumsum[49], 'b--', label='top 50 agents')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    plt.xlabel('PPO iteration')
    plt.ylabel('Sampling prob')
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/analyze_sampling_prob_{name}.png')
    plt.close()


def analyze_expanded(folders):
    # check the family size
    plt.figure()
    for folder in folders:
        family_size = defaultdict(int)
        for agent in os.listdir(f'unimals_100/{folder}/xml'):
            root = '-'.join(agent[:-4].split('-')[:8])
            family_size[root] += 1
        family_size = np.array(list(family_size.values()))
        family_size = np.sort(family_size)
        plt.plot(family_size, label=folder)
    plt.legend()
    plt.savefig('figures/check_expanded_family_size.png')
    plt.close()

    plt.figure()
    for folder in folders:
        depth = []
        for agent in os.listdir(f'unimals_100/{folder}/xml'):
            d = int((len(agent[:-4].split('-')) - 8) / 2)
            depth.append(d)
        plt.plot(np.bincount(depth), label=folder)
    plt.legend()
    plt.savefig('figures/check_expanded_depth.png')
    plt.close()


def analyze_GAE_score(folder, name, max_iter=1219):

    # load agents
    cfg.merge_from_file(f'{folder}/config.yaml')
    agents = cfg.ENV.WALKERS
    # agents = os.listdir('unimals_100/train/xml')
    # agents = [x[:-4] for x in agents]

    # get ACCEL scores
    potential_score = defaultdict(list)
    for i in range(max_iter):
        with open(f'{folder}/ACCEL_score/{i}.pkl', 'rb') as f:
            p, s = pickle.load(f)
        for j, agent in enumerate(agents):
            potential_score[agent].append(p[j])

    # get sampling prob
    iter_prob = defaultdict(list)
    for i in range(max_iter):
        with open(f'{folder}/iter_prob/{i}.pkl', 'rb') as f:
            p = pickle.load(f)
        for j, agent in enumerate(agents):
            iter_prob[agent].append(p[j])

    # get the agent's average training score after each iteration
    return_curve = defaultdict(list)
    with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
        results = json.load(f)
    for agent in agents:
        return_curve[agent] = results[agent]['reward']['reward'][:max_iter]

    # get the performance upper bound
    ST_scores = {}
    ST_folder = 'output2/ST_MLP_train_mutate_constant_lr'
    for agent in agents:
        with open(f'{ST_folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
            log = json.load(f)
        ST_scores[agent] = np.max(log[agent]['reward']['reward'])
    # ST_scores = [ST_scores[agent] for agent in agents]

    optimal_score = {}
    for agent in agents:
        optimal_score[agent] = max(np.max(ST_scores[agent]), np.max(results[agent]['reward']['reward']))

    os.makedirs(f'figures/analyze_GAE_regret/{name}', exist_ok=True)

    # regret as x axis
    for i in range(0, max_iter, 100):
        plt.figure()
        if i == 0:
            xdata = [optimal_score[agent] - np.mean(return_curve[agent][i:i+3]) for agent in agents]
        else:
            xdata = [optimal_score[agent] - np.mean(return_curve[agent][i-2:i+3]) for agent in agents]
        ydata = [potential_score[agent][i] for agent in agents]
        colors = np.clip([optimal_score[agent] for agent in agents], 0., 5000.)
        plt.scatter(xdata, ydata, s=10, c=colors)
        plt.xlabel('Regret')
        plt.ylabel('Proxy')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'figures/analyze_GAE_regret/{name}/regret_score_{i:04}.png')
        plt.close()

    for i in range(0, max_iter, 100):
        plt.figure()
        if i == 0:
            xdata = [optimal_score[agent] - np.mean(return_curve[agent][i:i+3]) for agent in agents]
        else:
            xdata = [optimal_score[agent] - np.mean(return_curve[agent][i-2:i+3]) for agent in agents]
        ydata = [iter_prob[agent][i] for agent in agents]
        colors = np.clip([optimal_score[agent] for agent in agents], 0., 5000.)
        plt.scatter(xdata, ydata, s=10, c=colors)
        plt.xlabel('Regret')
        plt.ylabel('Sampling prob')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'figures/analyze_GAE_regret/{name}/regret_prob_{i:04}.png')
        plt.close()


def plot_per_agent_training_curve(folder):
    cmap = plt.cm.get_cmap('viridis')
    with open(f'output/{folder}/1409/Unimal-v0_results.json', 'r') as f:
        results = json.load(f)
    plt.figure()
    del results['__env__']
    del results['fps']
    agents = list(results.keys())
    final_score = [np.array(results[agent]['reward']['reward'][-10:]).mean() for agent in agents]
    order = np.argsort(final_score)
    agents = [agents[i] for i in order]
    for i, agent in enumerate(agents):
        plt.plot(results[agent]['reward']['reward'], c=cmap(i/len(agents)))
    plt.tight_layout()
    plt.savefig(f'figures/per_agent_curve_{folder}.png')
    plt.close()

    plt.figure()
    plt.plot(np.array(final_score)[order])
    plt.savefig('figures/per_agent_score.png')
    plt.close()

# one subplot for each training robot
def plot_per_agent_training_curve_v2(folders):

    agents = [x[:-4] for x in os.listdir('unimals_100/train/xml')]

    results = {}
    for folder in folders:
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            results[folder] = json.load(f)
    
    plt.figure(figsize=(20, 20))
    for i, agent in enumerate(agents):
        plt.subplot(10, 10, i + 1)
        for folder in folders:
            plt.plot(results[folder][agent]['reward']['reward'], label=folder)
        plt.ylim(-100., 5500.)
    plt.legend()
    plt.savefig('figures/per_agent_train_curve.png')
    plt.close()



def find_root(folders):
    plt.figure()
    for folder in folders:
        root_size = defaultdict(int)
        with open(f'{folder}/walkers.json', 'r') as f:
            robots = json.load(f)
        for robot in robots:
            if 'mutate' not in robot:
                root_size[robot] += 1
            else:
                root = robot.split('mutate')[0][:-1]
                root_size[root] += 1
        root_size = np.sort(list(root_size.values()))
        print (root_size)
        plt.plot(root_size)
    plt.savefig('figures/check_root_size.png')
    plt.close()


def train_mutation_generalization_curve(folders):
    mutation_curve = defaultdict(dict)
    for folder in folders:
        for iteration in range(600, 5000, 600):
            with open(f'eval/ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout/ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout_1409_{folder}_cp_{iteration}.pkl', 'rb') as f:
                eval_results = pickle.load(f)
            for agent in eval_results:
                root = agent.split('mutate')[0][:-1]
                if agent not in mutation_curve[root]:
                    mutation_curve[root][agent] = [0.]
                mutation_curve[root][agent].append(np.mean(eval_results[agent][0]))
    
    plt.figure(figsize=(20, 20))
    for i, root in enumerate(mutation_curve):
        if i == 100:
            break
        plt.subplot(10, 10, i + 1)
        for agent in mutation_curve[root]:
            plt.plot(mutation_curve[root][agent])
        plt.ylim(-100, 5500)
    plt.savefig('figures/per_train_agent_mutation_curve.png')
    plt.close()


def twin_plot_train_curve_sample_prob(folders, name):

    os.makedirs(f'figures/analyze_UED/{name}', exist_ok=True)

    def score_to_prob(score):
        prob = np.clip(score, 0., 1.)
        prob /= prob.sum()
        return prob

    results, iter_prob, potential_score, staleness_score = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for folder in folders:
        cfg.merge_from_file(f'{folder}/config.yaml')
        # agents = cfg.ENV.WALKERS
        try:
            with open(f'{folder}/walkers_train.json', 'r') as f:
                agents = json.load(f)
        except:
            with open(f'{folder}/walkers.json', 'r') as f:
                agents = json.load(f)
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            results[folder] = json.load(f)
        data_seq = []
        for i in range(10000):
            try:
                with open(f'{folder}/iter_prob/{i}.pkl', 'rb') as f:
                    probs = pickle.load(f)
                    probs = np.concatenate([probs, np.zeros(len(agents) - len(probs))])
                    iter_prob[folder].append(probs)
            except:
                if i <= 30:
                    iter_prob[folder].append(np.concatenate([np.ones(100) / 100., np.zeros(len(agents) - 100)]))
                else:
                    break
        for i in range(10000):
            try:
                with open(f'{folder}/proxy_score/{i}.pkl', 'rb') as f:
                    data = pickle.load(f)
            except:
                if i <= 30:
                    potential_score[folder].append(np.zeros(len(agents)))
                    staleness_score[folder].append(np.zeros(len(agents)))
                    continue
                else:
                    break
            try:
                potential_score[folder].append(np.concatenate([data['LP'], np.zeros(len(agents) - len(data['LP']))]))
                # staleness_score[folder].append(np.concatenate([data['staleness'], np.zeros(len(agents) - len(data['staleness']))]))
            except:
                potential_score[folder].append(data[0])
                # staleness_score[folder].append(data[1])
        try:
            iter_prob[folder] = np.stack(iter_prob[folder])
            potential_score[folder] = np.stack(potential_score[folder])
            iter_prob[folder] = {agent: iter_prob[folder][:, i] for i, agent in enumerate(agents)}
            potential_score[folder] = {agent: potential_score[folder][:, i] for i, agent in enumerate(agents)}
            # staleness_score[folder] = np.stack(staleness_score[folder])
        except:
            continue

    # get ST performance
    # with open(f'mutate_400_upper_bound.json', 'r') as f:
    #     ST_score = json.load(f)

    for i, agent in enumerate(agents):
        # fig, ax1 = plt.subplots()
        x_bound = cfg.PPO.MAX_ITERS
        plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1)
        for folder in folders:
            # plt.plot(results[folder][agent]['return_ema'], label=folder)
            plt.plot(results[folder][agent]['reward']['reward'], label=folder)
        # plt.plot([0, x_bound], [ST_score[agent], ST_score[agent]], 'k--')
        plt.ylim(-50, 5500)
        plt.xlim(0, x_bound)
        # ax2 = ax1.twinx()
        plt.legend()
        plt.subplot(2, 2, 2)
        for folder in folders:
            if len(iter_prob[folder]) != 0:
                plt.plot(iter_prob[folder][agent], label=folder)
        plt.title('sample prob')
        plt.xlim(0, x_bound)
        # ax2.plot([0, twin_data.shape[0]], [1./400, 1./400], 'k--')
        # ax2.set_ylim(0, 0.1)
        plt.subplot(2, 2, 3)
        for folder in folders:
            if len(potential_score[folder]) != 0:
                plt.plot(potential_score[folder][agent], label=folder)
        plt.title('potential score')
        plt.xlim(0, x_bound)
        # plt.ylim(0., 0.2)
        # plt.subplot(2, 2, 4)
        # for folder in folders:
        #     if len(staleness_score[folder]) != 0:
        #         plt.plot(staleness_score[folder][:, i], label=folder)
        # plt.title('staleness score')
        # plt.xlim(0, x_bound)
        # plt.ylim(0, 5)
        plt.subplot(2, 2, 4)
        for folder in folders:
            plt.plot(np.cumsum(iter_prob[folder][agent]), label=folder)
        plt.title('cummulated sample prob')
        plt.xlim(0, x_bound)
        plt.savefig(f'figures/analyze_UED/{name}/{agent}.png')
        plt.close()


def analyze_GAE_proxy(folder, policy_name):

    os.makedirs(f'figures/check_value/{folder}/{policy_name}', exist_ok=True)

    cfg.merge_from_file(f'output/{folder}/config.yaml')
    model_path = f'output/{folder}/{policy_name}.pt'
    cfg.PPO.CHECKPOINT_PATH = model_path
    # cfg.ENV.WALKERS = []
    # cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    # do not include terminating on falling during evaluation
    # cfg.TERMINATE_ON_FALL = terminate_on_fall
    # cfg.DETERMINISTIC = deterministic
    set_cfg_options()
    ppo_trainer = PPO()
    cfg.PPO.NUM_ENVS = 16
    policy = ppo_trainer.agent
    # change to eval mode as we have dropout in the model
    policy.ac.eval()

    for agent in cfg.ENV.WALKERS:
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        obs = envs.reset()

        episode_return = np.zeros(cfg.PPO.NUM_ENVS)
        not_done = np.ones(cfg.PPO.NUM_ENVS)
        value_record = np.zeros([cfg.PPO.NUM_ENVS, 1000])
        reward_record = np.zeros([cfg.PPO.NUM_ENVS, 1000])
        episode_length = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)
        timeout_flag = np.zeros(cfg.PPO.NUM_ENVS)
        for t in range(2000):
            val, act, _, _, _ = policy.act(obs, return_attention=False, compute_val=True)
            value_record[:, t] = val.detach().cpu().numpy().ravel()
            obs, reward, done, infos = envs.step(act)
            reward_record[:, t] = reward.ravel()
            idx = np.where(done)[0]
            for i in idx:
                if not_done[i] == 1:
                    not_done[i] = 0
                    episode_return[i] = infos[i]['episode']['r']
                    episode_length[i] = t + 1
                    if "timeout" in infos[i]:
                        timeout_flag[i] = 1
            if not_done.sum() == 0:
                break
        
        # plot and compute GAE
        all_gae = []
        plt.figure(figsize=(32, 32))
        for i in range(cfg.PPO.NUM_ENVS):
            plt.subplot(4, 4, i + 1)
            values = value_record[i, :episode_length[i]]
            rewards = reward_record[i, :episode_length[i]]
            returns = np.zeros(episode_length[i])
            if timeout_flag[i]:
                returns[-1] = values[-1]
            else:
                returns[-1] = rewards[-1]
            for j in reversed(range(episode_length[i] - 1)):
                returns[j] = rewards[j] + cfg.PPO.GAMMA * returns[j + 1]
            # plt.plot(values, label='value pred')
            # plt.plot(returns, label='normalized return')
            plt.xlim(0, 1000)

            # compute GAE
            gamma, gae_lambda = cfg.PPO.GAMMA, cfg.PPO.GAE_LAMBDA
            # val: (T+1, P, 1), self.val: (T, P, 1) next_value: (P, 1)
            gae = np.zeros(episode_length[i])
            delta = np.zeros(episode_length[i])
            if timeout_flag[i]:
                gae[-1] = delta[-1] = 0.
            else:
                gae[-1] = delta[-1] = rewards[-1] - values[-1]
            for j in reversed(range(episode_length[i] - 1)):
                delta[j] = rewards[j] + gamma * values[j + 1] - values[j]
                gae[j] = delta[j] + gamma * gae_lambda * gae[j + 1]
            all_gae.append(gae)
            # plt.plot(gae, label='GAE')
            plt.plot(delta, label='delta')
            plt.title(f'R={int(episode_return[i])}, GAE={gae.mean():.2f}, PVL={np.maximum(gae, 0.).mean():.2f}, L1 value loss={np.abs(gae).mean():.2f}')

        plt.legend()
        plt.savefig(f'figures/check_value/{folder}/{policy_name}/{agent}.png')
        plt.close()
        # with open(f'figures/check_value/{folder}/{policy_name}/{agent}_GAE.pkl', 'wb') as f:
        #     pickle.dump(all_gae, f)


def compare_GAE_performance_correlation(folder, func):

    # load agents
    cfg.merge_from_file(f'output/{folder}/config.yaml')
    agents = cfg.ENV.WALKERS

    # # get the performance upper bound
    # ST_scores = {}
    # ST_folder = 'output/ST_MLP_train_mutate_constant_lr'
    # for agent in agents:
    #     with open(f'{ST_folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         log = json.load(f)
    #     ST_scores[agent] = np.array(log[agent]['reward']['reward'][-10:]).mean()
    # # ST_scores = [ST_scores[agent] for agent in agents]

    # get the agent's average training score after each iteration
    return_curve = defaultdict(list)
    with open(f'output/{folder}/Unimal-v0_results.json', 'r') as f:
        results = json.load(f)
    for agent in agents:
        return_curve[agent] = results[agent]['reward']['reward']

    plt.figure(figsize=(20, 10))
    for i, checkpoint in enumerate(range(600, 5400, 600)):
        if not os.path.exists(f'figures/check_value/{folder}/checkpoint_{checkpoint}'):
            continue
        plt.subplot(2, 4, i + 1)
        gae_scores = []
        for agent in agents:
            try:
                with open(f'figures/check_value/{folder}/checkpoint_{checkpoint}/{agent}_GAE.pkl', 'rb') as f:
                    gae = pickle.load(f)
                if func == 'absolute':
                    score = np.mean([np.abs(x).mean() for x in gae])
                elif func == 'positive':
                    score = np.mean([np.maximum(x, 0.).mean() for x in gae])
                gae_scores.append(score)
            except:
                gae_scores.append(np.nan)
        xdata = [np.mean(return_curve[agent][(checkpoint-3):(checkpoint+3)]) for agent in agents]
        ydata = gae_scores
        plt.scatter(xdata, ydata, s=5)
        plt.ylim(0., 1.)
        if i >= 4:
            plt.xlabel('train performance')
        if i in [0, 4]:
            plt.ylabel('GAE score')
        plt.title(f'iteration {checkpoint}')
    plt.savefig(f'figures/check_GAE_variant/{func}.png')
    plt.close()


def analyze_GAE_regret_correlation(model, name, optimal_score, fig_name):

    os.makedirs(f'figures/analyze_GAE_regret/{name}', exist_ok=True)

    with open(model, 'rb') as f:
        eval_results = pickle.load(f)
    
    agents = list(eval_results.keys())
    print ('eval agents num', len(agents))
    agents = list(set(agents).intersection(set(optimal_score.keys())))
    print ('merged agents num', len(agents))

    regret = {}
    positive_GAE, absolute_GAE, avg_GAE = {}, {}, {}
    for agent in agents:
        score, _, GAE = eval_results[agent]
        generalization_score = np.mean(score)
        regret[agent] = optimal_score[agent] - generalization_score
        positive_GAE[agent] = np.mean([np.maximum(x, 0.).mean() for x in GAE])
        absolute_GAE[agent] = np.mean([np.abs(x).mean() for x in GAE])
        avg_GAE[agent] = np.mean([x.mean() for x in GAE])

    plt.figure()
    colors = np.clip([optimal_score[agent] for agent in agents], 0., 5000.)
    plt.scatter([regret[agent] for agent in agents], [avg_GAE[agent] for agent in agents], s=5, c=colors)
    plt.colorbar(label='ST optimal score')
    plt.xlabel('regret')
    plt.ylabel('average GAE')
    plt.savefig(f'figures/analyze_GAE_regret/{name}/GAE_{fig_name}.png')
    plt.close()

    plt.figure()
    plt.scatter([regret[agent] for agent in agents], [positive_GAE[agent] for agent in agents], s=5, c=colors)
    plt.colorbar(label='ST optimal score')
    plt.xlabel('regret')
    plt.ylabel('positive value loss')
    plt.savefig(f'figures/analyze_GAE_regret/{name}/PVL_{fig_name}.png')
    plt.close()

    plt.figure()
    plt.scatter([regret[agent] for agent in agents], [absolute_GAE[agent] for agent in agents], s=5, c=colors)
    plt.colorbar(label='ST optimal score')
    plt.xlabel('regret')
    plt.ylabel('L1 value loss')
    plt.savefig(f'figures/analyze_GAE_regret/{name}/L1_value_loss_{fig_name}.png')
    plt.close()


def print_iter_prob(folder):
    for i in range(1000):
        try:
            with open(f'{folder}/1409/iter_prob/{i}.pkl', 'rb') as f:
                prob = pickle.load(f)
            print (np.sort(prob))
        except:
            break


def get_upper_bound(folders):
    
    results = {}
    for folder in folders:
        with open(f'{folder}/Unimal-v0_results.json', 'r') as f:
            results[folder] = json.load(f)
    
    agents = [x[:-4] for x in os.listdir('data/train_mutate_400/xml')]
    
    upper_bound = {}
    for agent in agents:
        upper_bound[agent] = max([np.max(results[folder][agent]['reward']['reward']) for folder in folders])
    with open('upper_bound_train_mutate_400_wo_height_check.json', 'w') as f:
        json.dump(upper_bound, f)


def analyze_mutation_root(folders, name):
    mutation_num = {}
    for folder in folders:
        tree_size = defaultdict(int)
        for xml in os.listdir(f'{folder}/xml'):
            root = xml[:-4].split('-mutate-')[0]
            if len(xml[:-4].split('-mutate-')) > 1 and len(xml[:-4].split('-mutate-')[1]) == 1:
                root = root + '-mutate-' + xml[:-4].split('-mutate-')[1]
            tree_size[root] += 1
        total_robot_num = len(os.listdir(f'{folder}/xml'))
        for root in tree_size:
            tree_size[root] /= total_robot_num
        mutation_num[folder] = tree_size
    
    plt.figure()
    for folder in folders:
        data = list(mutation_num[folder].values())
        data.sort()
        plt.plot(data, label=folder)
    plt.legend()
    plt.savefig(f'figures/generation_root_distribution_{name}.png')
    plt.close()



if __name__ == '__main__':

    folders = [
        'data/train_generation_LP', 
        'data/train_generation_LP_start_30', 
        'data/train_generation_uniform_start_30', 
    ]
    name = 'init_train'
    analyze_mutation_root(folders, name)

    folders = [
        'data/train_mutate_400_generation_uniform_start_30', 
        'data/train_mutate_400_generation_uniform_balanced', 
        'data/train_mutate_400_generation_LP', 
        'data/train_mutate_400_generation_LP_start_30', 
        'data/train_mutate_400_generation_LP_new', 
        'data/train_mutate_400_generation_LP_new_balanced', 
    ]
    name = 'init_mutate_400'
    analyze_mutation_root(folders, name)

    folders = {
        'ft_400M_env_256_default_100_maximin_KL_5_wo_PE+dropout': 100, 
        'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
        'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
        # 'ft_random_100_uniform_sample_KL_5_wo_PE+dropout': 600, 
        # 'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
        'ft_1000M_env_256_random_10k_uniform_sample_KL_5_wo_PE+dropout': 100, 
        'ft_1000M_env_256_random_100k_uniform_sample_KL_5_wo_PE+dropout': 100, 
    }
    names = {
        'ft_400M_env_256_default_100_maximin_KL_5_wo_PE+dropout': 'default_100, 256 envs', 
        'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate 400, 256 envs', 
        'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, 256 envs', 
        # 'ft_random_100_uniform_sample_KL_5_wo_PE+dropout': 'random_100, 32 envs', 
        # 'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'random_1000, 256 envs', 
        'ft_1000M_env_256_random_10k_uniform_sample_KL_5_wo_PE+dropout': 'random 10k, 256 envs, 1000M steps', 
        'ft_1000M_env_256_random_100k_uniform_sample_KL_5_wo_PE+dropout': 'random 100k, 256 envs, 1000M steps', 
    }
    suffix = 'mutate+random'
    # generalization_curve(folders, names, 'random_100_holdout', suffix, plot_each_seed=False)
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)
    generalization_curve(folders, names, 'random_100_above_1000', suffix, plot_each_seed=False)

    folders = {
        'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
        'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_env_256_init_train_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_train_generation_LP_curation_uniform_start_30_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_default_100_start_30_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_mutate_400_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_mutate_400_start_30_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 100, 
        'ft_400M_init_mutate_400_generation_uniform_balanced_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_mutate_400_start_30_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_mutate_400_generation_LP_new_curation_uniform_KL_5_wo_PE+dropout': 100, 
        'ft_400M_init_mutate_400_generation_LP_new_balanced_curation_uniform_KL_5_wo_PE+dropout': 100, 
        # 'ft_400M_init_base_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 100, 
    }
    names = {
        'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate 400, uniform sample', 
        'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate 1000, uniform sample', 
        # 'ft_400M_env_256_init_train_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 'LP generation, init train', 
        # 'ft_400M_init_train_generation_LP_curation_uniform_start_30_KL_5_wo_PE+dropout': 'LP generation, init train', 
        # 'ft_400M_init_default_100_start_30_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 'uniform generation, init train', 
        # 'ft_400M_init_mutate_400_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 'LP generation, init mutate 400', 
        # 'ft_400M_init_mutate_400_start_30_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 'uniform generation, init mutate 400', 
        'ft_400M_init_mutate_400_generation_uniform_balanced_curation_uniform_KL_5_wo_PE+dropout': 'uniform generation (balanced), init mutate 400', 
        # 'ft_400M_init_mutate_400_start_30_generation_LP_curation_uniform_KL_5_wo_PE+dropout': 'LP generation, init mutate 400', 
        # 'ft_400M_init_mutate_400_generation_LP_new_curation_uniform_KL_5_wo_PE+dropout': 'LP generation new, init mutate 400', 
        'ft_400M_init_mutate_400_generation_LP_new_balanced_curation_uniform_KL_5_wo_PE+dropout': 'LP generation new (balanced), init mutate 400', 
        # 'ft_400M_init_base_generation_uniform_curation_uniform_KL_5_wo_PE+dropout': 'uniform generation, init base', 
    }
    suffix = 'generation_LP'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)
    generalization_curve(folders, names, 'random_100_holdout', suffix, plot_each_seed=False)

    # folder = 'output/ft_MLP_HN_IO_onehot_KL_5_wo_PE+dropout'
    # print_iter_prob(folder)

    # folders = [
    #     'output/ft_400M_mutate_400_wo_height_check_env_256_uniform_sample_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_mutate_400_wo_height_check_env_256_curation_regret_delta_0.1_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_mutate_400_wo_height_check_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout/1409', 
    # ]
    # get_upper_bound(folders)

    # folders = {
    #     'ft_random_100_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_random_100_curation_regret_KL_5_wo_PE+dropout': 600, 
    #     'ft_random_100_curation_regret_alpha_0.1_KL_5_wo_PE+dropout': 600, 
    #     'ft_random_100_curation_ratio_regret_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_random_1000_curation_regret_env_256_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_random_100_uniform_sample_KL_5_wo_PE+dropout': 'random 100, uniform sample', 
    #     'ft_random_100_curation_regret_KL_5_wo_PE+dropout': 'random 100, regret curation', 
    #     'ft_random_100_curation_regret_alpha_0.1_KL_5_wo_PE+dropout': 'random 100, regret curation, alpha = 0.1', 
    #     'ft_random_100_curation_ratio_regret_KL_5_wo_PE+dropout': 'random 100, relative regret curation', 
    #     'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'random 1000, uniform sample', 
    #     'ft_400M_random_1000_curation_regret_env_256_KL_5_wo_PE+dropout': 'random 1000, regret curation', 
    # }
    # suffix = 'curation_random_100+1000'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)
    # generalization_curve(folders, names, 'random_100_holdout', suffix, plot_each_seed=False)

    # shorten the old eval result name
    # folders = [
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # for folder in folders:
    #     files = os.listdir(f'eval/{folder}')
    #     for f in files:
    #         if '1409' in f:
    #             flag = '1409'
    #         else:
    #             flag = '1410'
    #         if 'test' in f or 'random_100_holdout' in f:
    #             new_f = flag + f.split(flag)[1]
    #             os.system(f'mv eval/{folder}/{f} eval/{folder}/{new_f}')
    #         else:
    #             os.system(f'rm eval/{folder}/{f}')

    # add seed number to eval result name
    # folders = [
    #     'ft_400M_mutate_400_curation_LP_env_256_KL_5_wo_PE+dropout', 
    #     'ft_400M_mutate_1000_curation_LP_env_256_KL_5_wo_PE+dropout', 
    #     'ft_400M_random_1000_curation_LP_env_256_KL_5_wo_PE+dropout', 
    # ]
    # for folder in folders:
    #     files = os.listdir(f'eval/{folder}')
    #     for f in files:
    #         os.system(f'mv eval/{folder}/{f} eval/{folder}/{f[5:]}')

    # folders = {
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_400_curation_LP_env_256_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_curation_LP_env_256_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_random_1000_curation_LP_env_256_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform', 
    #     'ft_400M_mutate_400_curation_LP_env_256_KL_5_wo_PE+dropout': 'mutate_400, LP curation', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform', 
    #     'ft_400M_mutate_1000_curation_LP_env_256_KL_5_wo_PE+dropout': 'mutate_1000, LP curation', 
    #     'ft_400M_random_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'random_1000, uniform', 
    #     'ft_400M_random_1000_curation_LP_env_256_KL_5_wo_PE+dropout': 'random_1000, LP curation', 
    # }
    # suffix = 'curation_LP'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)
    # generalization_curve(folders, names, 'random_100_holdout', suffix, plot_each_seed=False)

    # folders = [
    #     'baselines/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_mutate_400_curation_LP_env_256_KL_5_wo_PE+dropout/1409', 
    #     'baselines/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1410', 
    # ]
    # name = 'mutate_400_curation_LP'
    # twin_plot_train_curve_sample_prob(folders, name)

    # upper_bound = {}

    # optimal_score = {}
    # for agent in os.listdir('output2/ST_MLP_test'):
    #     with open(f'output2/ST_MLP_test/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     optimal_score[agent] = np.max(results[agent]['reward']['reward'])
    # upper_bound['test'] = optimal_score

    # optimal_score = {}
    # for agent in os.listdir('output2/ST_MLP_random_1000_constant_lr'):
    #     with open(f'output2/ST_MLP_random_1000_constant_lr/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     optimal_score[agent] = np.max(results[agent]['reward']['reward'])
    # upper_bound['random_100'] = optimal_score

    # # optimal_score = {}
    # # for name in os.listdir('unimals_100/train/xml'):
    # #     agent = name[:-4]
    # #     with open(f'output2/ST_MLP_train_mutate_constant_lr/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    # #         results = json.load(f)
    # #     optimal_score[agent] = np.max(results[agent]['reward']['reward'])
    # # upper_bound['train'] = optimal_score

    # optimal_score = {}
    # with open('output/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409/Unimal-v0_results.json', 'r') as f:
    #     MT_results = json.load(f)
    # for name in os.listdir('unimals_100/mutate_300/xml'):
    #     agent = name[:-4]
    #     with open(f'output2/ST_MLP_train_mutate_constant_lr/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     ST_score = np.max(results[agent]['reward']['reward'])
    #     MT_score = np.max(MT_results[agent]['reward']['reward'])
    #     optimal_score[agent] = max(ST_score, MT_score)
    # upper_bound['mutate_300'] = optimal_score

    # optimal_score = {}
    # with open('output/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409/Unimal-v0_results.json', 'r') as f:
    #     MT_results = json.load(f)
    # for name in os.listdir('unimals_100/train_mutate_400/xml'):
    #     agent = name[:-4]
    #     with open(f'output2/ST_MLP_train_mutate_constant_lr/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         results = json.load(f)
    #     ST_score = np.max(results[agent]['reward']['reward'])
    #     MT_score = np.max(MT_results[agent]['reward']['reward'])
    #     optimal_score[agent] = max(ST_score, MT_score)
    # upper_bound['train_mutate_400'] = optimal_score
    # with open('mutate_400_upper_bound.json', 'w') as f:
    #     json.dump(optimal_score, f)

    # # folder = 'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout'
    # # name = '400M_mutate_400_env_256_uniform_sample'
    # # for test_set in ['test', 'random_100', 'train_mutate_400']:
    # #     for iteration in range(100, 700, 100):
    # #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}.pkl'
    # #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}')

    # # folder = 'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout'
    # # name = '400M_mutate_1000_env_256_uniform_sample'
    # # for test_set in ['test', 'random_100', 'mutate_300']:
    # #     for iteration in range(100, 500, 100):
    # #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}.pkl'
    # #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}')

    # folder = 'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout'
    # name = '400M_mutate_1000_env_256_curation_PVL'
    # for test_set in ['mutate_300']:
    #     for iteration in range(100, 500, 100):
    #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}.pkl'
    #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}')
    #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}_wo_reward_update.pkl'
    #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}_no_reward_norm')

    # folder = 'ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout'
    # name = '400M_mutate_400_env_256_curation_PVL'
    # for test_set in ['train_mutate_400']:
    #     for iteration in range(100, 700, 100):
    #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}.pkl'
    #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}')
    #         model = f'eval/{folder}/{folder}_1409_{test_set}_cp_{iteration}_wo_reward_update.pkl'
    #         analyze_GAE_regret_correlation(model, name, upper_bound[test_set], f'{test_set}_{iteration}_no_reward_norm')

    # folder = 'output/ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout/1409'
    # name = '400M_mutate_400_env_256_curation_L1_value_loss'
    # analyze_GAE_score(folder, name, max_iter=600)

    # folders = [
    #     'output/ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout/1409', 
    #     # 'output/ft_400M_mutate_400_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_mutate_400_env_256_curation_LP_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409', 
    # ]
    # name = 'curation_PVL'
    # twin_plot_train_curve_sample_prob(folders, name)

    # folder = 'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409'
    # policy_name = 'checkpoint_600'
    # analyze_GAE_proxy(folder, policy_name)

    # folder = 'output/ft_400M_mutate_400_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout/1409'
    # name = '400M_mutate_400_env_256_curation_L1_value_loss_delta_0.1'
    # analyze_GAE_score(folder, name, max_iter=400)

    # folder = 'output/ft_400M_mutate_400_env_256_curation_LP_KL_5_wo_PE+dropout/1409'
    # name = 'curation_LP_400M_mutate_400_env_256'
    # analyze_L1_value_loss(folder, name, max_iter=5000)

    # for agent in os.listdir('unimals_100/subset_10_0/xml'):
    #     name = agent[:-4]
    #     os.system('mkdir unimals_100/subset_10_0/unimal_init')
    #     os.system(f'cp unimals_100/train/unimal_init/{name}.pkl unimals_100/subset_10_0/unimal_init/')

    # folder = 'ft_baseline_KL_5_wo_PE+dropout'
    # plot_per_agent_training_curve(folder)

    # folder = 'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout/1409'
    # # for checkpoint in [600, 1200, 1800, 2400]:
    # # for checkpoint in [3000, 3600, 4200, 4800]:
    # #     policy_name = f'checkpoint_{checkpoint}'
    # #     analyze_GAE_rpoxy(folder, policy_name)
    # compare_GAE_performance_correlation(folder, 'absolute')
    # compare_GAE_performance_correlation(folder, 'positive')

    # folders = [
    #     'output/ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout/1410', 
    # ]
    # find_root(folders)
    # plot_per_agent_training_curve_v2(folders)

    # folders = [
    #     'train_remove_level_1', 
    #     'train_remove_level_2', 
    #     'train_remove_level_3', 
    # ]
    # train_mutation_generalization_curve(folders)

    # cfg.merge_from_file(f'{folder}/{seed}/config.yaml')
    # agents = cfg.WALKERS

    # ST_scores = {}
    # ST_folder = 'output/ST_MLP_train_mutate_constant_lr'
    # agents = os.listdir(ST_folder)
    # for agent in agents:
    #     with open(f'{ST_folder}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
    #         log = json.load(f)
    #     ST_scores[agent] = np.array(log[agent]['reward']['reward'][-10:]).mean()
    # # ST_scores = [ST_scores[agent] for agent in agents]

    # folders = [
    #     'output/ft_train_mutate_UED_KL_5_wo_PE+dropout', 
    #     'output/ft_train_mutate_KL_5_wo_PE+dropout', 
    #     'output/ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # for folder in folders:
    #     check_env_sample_freq(folder, ST_scores)

    # folder = 'output/ft_400M_train_KL_5_wo_PE+dropout/1410'
    # test_set = 'unimals_100/train'
    # evaluate_checkpoint(folder, test_set, interval=600)

    # get_context()
    # analyze_modular_obs_scale('walker')
    # analyze_train_robot_num()
    # compare_evaluation_results()
    # folders = [
    #     'output/ST_MLP_joint_angle_300_constant_lr', 
    #     'output/ST_MLP_train_mutate_constant_lr', 
    # ]
    # plot_ST_score(folders, 'compare')

    # curation with PVL
    # folders = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_400_curation_PVL_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_400_curation_PVL_staleness_0.3_KL_5_wo_PE+dropout': 200, 
    # }
    # names = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 'default_100, maximin', 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform', 
    #     'ft_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform', 
    #     'ft_mutate_400_curation_PVL_KL_5_wo_PE+dropout': 'mutation_400, PVL curation', 
    #     'ft_mutate_400_curation_PVL_staleness_0.3_KL_5_wo_PE+dropout': 'mutation_400, PVL curation, staleness=0.3', 
    # }
    # suffix = 'curation_PVL_100M'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #      'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 'mutation_1000, PVL curation, 256 envs', 
    # }
    # suffix = 'curation_PVL_400M_mutate_100_v2'
    # generalization_curve(folders, names, 'train_mutate_100_v2', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_400_wo_height_check_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_400_wo_height_check_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_400_wo_height_check_env_256_curation_regret_delta_0.1_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_400_wo_height_check_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform curation', 
    #     'ft_400M_mutate_400_wo_height_check_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout': 'mutate_400, PVL curation', 
    #     'ft_400M_mutate_400_wo_height_check_env_256_curation_regret_delta_0.1_KL_5_wo_PE+dropout': 'mutate_400, regret curation', 
    # }
    # suffix = 'height_check'
    # generalization_curve(folders, names, 'test', suffix, height_check=True, plot_each_seed=False)
    # name = 'wo_height_check_curation'
    # twin_plot_train_curve_sample_prob([f'output/{x}/1409' for x in list(folders.keys())], name)

    # folders = {
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout': 100, 
    #     # 'ft_400M_mutate_400_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout': 100, 
    #     # 'ft_400M_mutate_400_env_256_curation_PVL_delta_0.3_KL_5_wo_PE+dropout': 100, 
    #     # 'ft_400M_mutate_400_env_256_curation_true_PVL_KL_5_wo_PE+dropout': 100, 
    #     # 'ft_400M_mutate_400_env_256_curation_regret_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 256 envs', 
    #     'ft_400M_mutate_400_env_256_curation_PVL_KL_5_wo_PE+dropout': 'mutation_400, L1 value loss curation, 256 envs', 
    #     # 'ft_400M_mutate_400_env_256_curation_PVL_delta_0.1_KL_5_wo_PE+dropout': 'mutation_400, L1 value loss curation, 256 envs, delta=0.1', 
    #     # 'ft_400M_mutate_400_env_256_curation_PVL_delta_0.3_KL_5_wo_PE+dropout': 'mutation_400, L1 value loss curation, 256 envs, delta=0.3', 
    #     # 'ft_400M_mutate_400_env_256_curation_true_PVL_KL_5_wo_PE+dropout': 'mutation_400, PVL curation, 256 envs', 
    #     # 'ft_400M_mutate_400_env_256_curation_regret_KL_5_wo_PE+dropout': 'mutation_400, regret curation, 256 envs', 
    #      'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 'mutation_1000, L1 value loss curation, 256 envs', 
    # }
    # suffix = 'curation_PVL_400M'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout': 'mutation_1000, PVL curation, 256 envs', 
    # }
    # suffix = 'curation_PVL_400M'
    # generalization_curve(folders, names, 'mutate_300', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_1000M_random_1000_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_train_mutate_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_train_KL_5_wo_PE+dropout': 600, 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 200, 
    # }
    # names = {
    #     'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 1000M', 
    #     'ft_400M_train_mutate_KL_5_wo_PE+dropout': 'mutate_400, maximin, 400M', 
    #     'ft_400M_train_KL_5_wo_PE+dropout': 'default_100, maximin, 400M', 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 'mutate_400, minimax regret, 100M', 
    #     'ft_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 100M', 
    # }

    # folders = [
    #     'output/ft_train_mutate_recheck_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_train_mutate_recheck_KL_5_wo_PE+dropout/1410', 
    #     'output/ft_train_mutate_UED_recheck_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_train_mutate_UED_recheck_KL_5_wo_PE+dropout/1410', 
    # ]
    # check_UED_trend(folders)

    # folders = [
    #     'train_debug_400M_1409', 
    #     'train_debug_400M_1410', 
    #     'train_expand_immediate_children_1409', 
    # ]
    # analyze_expanded(folders)

    # folders = [
    #     'output/ft_debug_train_mutate_accel_wo_generation_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_debug_train_accel_wo_generation_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_debug_baseline_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_debug_baseline_UED_KL_5_wo_PE+dropout/1409', 
    #     'output/ft_debug_baseline_uniform_sample_KL_5_wo_PE+dropout/1409', 
    # ]
    # names = [
    #     'ACCEL (mutate_400)', 
    #     'ACCEL (default_100)', 
    #     'maximin', 
    #     'regret', 
    #     'uniform', 
    # ]
    # analyze_GAE_score(folders[0], names[0])
    # check_ACCEL(folders)
    # for folder, name in zip(folders, names):
        # analyze_L1_value_loss(folder, name)

    # folders = {
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_debug_accel_wo_selection_KL_5_wo_PE+dropout': 200, 
    # }
    # names = {
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 100M', 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 100M', 
    #     'ft_debug_accel_wo_selection_KL_5_wo_PE+dropout': 'expand, uniform, 100M', 
    # }
    # suffix = '100M_ACCEL_generation'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_debug_train_accel_wo_generation_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_debug_train_mutate_accel_wo_generation_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_debug_mutate_1000_PVL_KL_5_wo_PE+dropout': 200, 
    # }
    # names = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 'default_100, maximin, 100M', 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 100M', 
    #     'ft_debug_train_accel_wo_generation_KL_5_wo_PE+dropout': 'default_100, positive value loss, 100M', 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 'mutate_400, minimax regret, 100M', 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 100M', 
    #     'ft_debug_train_mutate_accel_wo_generation_KL_5_wo_PE+dropout': 'mutate_400, positive value loss, 100M', 
    #     'ft_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 100M', 
    #     'ft_debug_mutate_1000_PVL_KL_5_wo_PE+dropout': 'mutate_1000, PVL, 100M', 
    # }
    # suffix = '100M_ACCEL_curation'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_UED_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 200, 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_400_UED_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_400_KL_5_wo_PE+dropout': 200, 
    #     'ft_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 200, 
    # }
    # names = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 'default_100, maximin, 100M', 
    #     'ft_baseline_UED_KL_5_wo_PE+dropout': 'default_100, minimax regret, 100M', 
    #     'ft_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 100M', 
    #     'ft_train_mutate_KL_5_wo_PE+dropout': 'mutate_400_old, maximin, 100M', 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout': 'mutate_400_old, minimax regret, 100M', 
    #     'ft_train_mutate_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400_old, uniform, 100M', 
    #     'ft_mutate_400_UED_KL_5_wo_PE+dropout': 'mutate_400, minimax regret, 100M', 
    #     'ft_mutate_400_KL_5_wo_PE+dropout': 'mutate_400, maximin, 100M', 
    #     'ft_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 100M'
    # }
    # suffix = '100M_UED_more'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_debug_1000M_mutate_1000_PVL_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_generate_validation_curate_uniform_KL_5_wo_PE+dropout': 600, 
    # }
    # names = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 400M', 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 400M', 
    #     # 'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 1000M', 
    #     # 'ft_debug_1000M_mutate_1000_PVL_KL_5_wo_PE+dropout': 'mutate_1000, PVL, 1000M', 
    #     'ft_400M_generate_validation_curate_uniform_KL_5_wo_PE+dropout': 'expand by validation, 400M', 
    # }
    # suffix = 'expand_validation_400M'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 600, 
    # }
    # names = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 400M', 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 400M', 
    #     'ft_1000M_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 1000M', 
    # }
    # suffix = ''
    # per_agent_generalization_curve(folders, names, 'test', suffix)

    # folders = {
    #     # 'ft_400M_train_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_baseline_UED_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_train_mutate_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_train_mutate_UED_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    # }
    # names = {
    #     # 'ft_400M_train_KL_5_wo_PE+dropout': 'default_100, maximin, 400M', 
    #     # 'ft_400M_baseline_UED_KL_5_wo_PE+dropout': 'default_100, minimax regret, 400M', 
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 400M', 
    #     # 'ft_400M_train_mutate_KL_5_wo_PE+dropout': 'mutate_400, maximin, 400M', 
    #     # 'ft_400M_train_mutate_UED_KL_5_wo_PE+dropout': 'mutate_400, minimax regret, 400M', 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 400M', 
    # }
    # suffix = '400M_UED'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_debug_400M_accel_wo_selection_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_expand_immediate_children+uniform_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout': 600, 
    # }
    # names = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform, 400M', 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 400M', 
    #     'ft_debug_400M_accel_wo_selection_KL_5_wo_PE+dropout': 'expanded, uniform, 400M', 
    #     'ft_400M_expand_immediate_children+uniform_KL_5_wo_PE+dropout': 'expanded, immediate children, 400M', 
    #     'ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout': 'expanded, immediate children, threshold=2000', 
    # }
    # suffix = '400M_ACCEL_generation'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_mutate_400_env_64_uniform_sample_KL_5_wo_PE+dropout': 300, 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_uniform_sample_KL_5_wo_PE+dropout': 300, 
    #     # 'ft_400M_mutate_400_env_64_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout': 300, 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout': 300, 
    #     # 'ft_400M_mutate_400_env_128_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_mutate_400_curation_LP_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_mutate_1000_curation_LP_KL_5_wo_PE+dropout': 600, 
    #     # 'ft_400M_mutate_400_env_256_curation_LP_KL_5_wo_PE+dropout': 100, 
    #     # 'ft_400M_mutate_1000_env_256_curation_LP_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_expand_validation_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_expand_validation_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform', 
    #     # 'ft_400M_mutate_400_env_64_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 64 envs', 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 64 envs, batchsize=10240', 
    #     # 'ft_400M_mutate_400_env_64_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 64 envs, no falling reset', 
    #     # 'ft_400M_mutate_400_env_64_batchsize_10240_no_falling_reset_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 64 envs, batchsize=10240, no falling reset', 
    #     # 'ft_400M_mutate_400_env_128_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 128 envs', 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 32 envs', 
    #     # 'ft_400M_mutate_400_curation_LP_KL_5_wo_PE+dropout': 'mutate_400, LP, 32 envs', 
    #     # 'ft_400M_mutate_1000_curation_LP_KL_5_wo_PE+dropout': 'mutate_1000, LP, 32 envs', 
    #     # 'ft_400M_mutate_400_env_256_curation_LP_KL_5_wo_PE+dropout': 'mutate_400, LP, 256 envs', 
    #     # 'ft_400M_mutate_1000_env_256_curation_LP_KL_5_wo_PE+dropout': 'mutate_1000, LP, 256 envs', 
    #     'ft_400M_expand_validation_uniform_sample_KL_5_wo_PE+dropout': 'expand, 32 envs', 
    #     'ft_400M_expand_validation_env_256_uniform_sample_KL_5_wo_PE+dropout': 'expand, 256 envs', 
    # }
    # suffix = '400M_check_env_num'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)
    # # batch_size = [16 * 8 * i for i in [1, 2, 1, 4, 8]]
    # # env_num = [32, 64, 64, 128, 256]
    # # # plot_training_stats(list(folders.keys()), names=[names[key] for key in names], batch_size=batch_size, env_num=env_num, kl_threshold=[0.05 for _ in batch_size], prefix='400M_env_num')

    # folders = {
    #     'ft_400M_train_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_train_KL_5_wo_PE+dropout': 'mutate_400, uniform', 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    # }
    # suffix = '400M'
    # generalization_curve(folders, names, 'joint_angle_100', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_expand_immediate_children+uniform_KL_5_wo_PE+dropout': 600, 
    # }
    # names = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform', 
    #     'ft_400M_mutate_400_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform', 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 32 envs', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_expand_immediate_children+uniform_KL_5_wo_PE+dropout': 'expand, uniform, 32 envs', 
    # }
    # suffix = 'random_100_400M'
    # generalization_curve(folders, names, 'random_100', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs, FA', 
    #     'ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs, HN+FA', 
    # }
    # suffix = '400M_mutate_1000_HN+FA'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs, FA', 
    #     'ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs, HN+FA', 
    # }
    # suffix = '400M_mutate_1000_HN+FA'
    # generalization_curve(folders, names, 'random_100', suffix, plot_each_seed=False)

    # folders = [
    #     'ft_400M_train_KL_5_wo_PE+dropout', 
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout', 
    #     'ft_400M_separatePE_uniform_sample_KL_5_wo_PE+dropout', 
    # ]
    # plot_training_stats(folders, prefix='400M_separate_PE')

    # folders = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 600, 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 100, 
    # }
    # names = {
    #     'ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout': 'default_100, uniform', 
    #     'ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_400, uniform, 256 envs', 
    #     'ft_400M_mutate_1000_env_256_uniform_sample_KL_5_wo_PE+dropout': 'mutate_1000, uniform, 256 envs', 
    # }
    # suffix = '1409'
    # generalization_curve(folders, names, 'train_remove_level_1', suffix, plot_each_seed=False, seeds=[1409])
    # generalization_curve(folders, names, 'train_remove_level_2', suffix, plot_each_seed=False, seeds=[1409])
    # generalization_curve(folders, names, 'train_remove_level_3', suffix, plot_each_seed=False, seeds=[1409])
    # per_agent_generalization_curve_v2(folders, names, 'train_remove_level_1', suffix)

    # folders = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_env_64_timestep_1280_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_env_128_timestep_640_KL_5_wo_PE+dropout': 200, 
    #     'ft_baseline_env_64_KL_5_wo_PE+dropout': 100, 
    #     'ft_baseline_env_128_KL_5_wo_PE+dropout': 50, 
    # }
    # names = {
    #     'ft_baseline_KL_5_wo_PE+dropout': 'default_100, maximin', 
    #     'ft_baseline_env_64_timestep_1280_KL_5_wo_PE+dropout': 'default_100, maximin, 64 envs, 1280 timesteps', 
    #     'ft_baseline_env_128_timestep_640_KL_5_wo_PE+dropout': 'default_100, maximin, 126 envs, 640 timesteps', 
    #     'ft_baseline_env_64_KL_5_wo_PE+dropout': 'default_100, maximin, 64 envs', 
    #     'ft_baseline_env_128_KL_5_wo_PE+dropout': 'default_100, maximin, 128 envs', 
    # }
    # suffix = '100M_check_env_num'
    # generalization_curve(folders, names, 'test', suffix, plot_each_seed=False)

    # folders = [
    #     'ft_train_mutate_KL_5_wo_PE+dropout/1409', 
    #     'ft_train_mutate_KL_5_wo_PE+dropout/1410', 
    #     'ft_train_mutate_UED_KL_5_wo_PE+dropout/1409', 
    # ]
    # ST_folder = 'ST_MLP_train_mutate_constant_lr'
    # for MT_folder in folders:
    #     suffix = MT_folder.replace('/', '_')
    #     compare_MT_ST_v2(ST_folder, MT_folder, suffix)

    # ST_folder = 'ST_MLP_train_mutate_constant_lr'
    # MT_eval_file = 'ft_train_mutate_v2_uniform_sample_KL_5_wo_PE+dropout_1409_train_mutate'
    # suffix = 'mutate_v1'
    # compare_MT_ST(ST_folder, MT_eval_file, suffix)

    # agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/random_1000/xml')]
    # num_agent = 400
    # agent_names = agent_names[:num_agent]
    # os.makedirs(f'unimals_100/random_{num_agent}', exist_ok=True)
    # os.makedirs(f'unimals_100/random_{num_agent}/xml', exist_ok=True)
    # os.makedirs(f'unimals_100/random_{num_agent}/metadata', exist_ok=True)
    # for agent in agent_names:
    #     os.system(f'cp unimals_100/random_1000/xml/{agent}.xml unimals_100/random_{num_agent}/xml/')
    #     os.system(f'cp unimals_100/random_1000/metadata/{agent}.json unimals_100/random_{num_agent}/metadata/')

    # compare policy divergence
    # agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/test/xml')][:40]
    # record = []
    # for agent in agent_names:
    #     source_folder = f'output/ST_test_MLP/{agent}/1409'
    #     # target_folder = 'output/ft_baseline_KL_5_wo_PE+dropout/1409'
    #     target_folder = f'output/ST_test_MLP/{agent}/1410'
    #     agent_path = f'unimals_single_task/{agent}'
    #     divergence = compare_policy_divergence(source_folder, target_folder, agent_path)
    #     print (agent, divergence)
        # compare performance
        # with open(f'output/ST_test_MLP/{agent}/1409/Unimal-v0_results.json', 'r') as f:
        #     log = json.load(f)
        # ST_score = np.array(log[agent]['reward']['reward'][-10:]).mean()
        # eval_file_name = '_'.join(target_folder.split('/')[1:])
        # with open(f'eval/{eval_file_name}_test.pkl', 'rb') as f:
        #     eval_results = pickle.load(f)
        # MT_score = np.array(eval_results[agent][0]).mean()

        # record.append([divergence, ST_score, MT_score])
        # with open('ST_MT_test_policy_divergence.pkl', 'wb') as f:
        #     pickle.dump(record, f)

    # with open('ST_MT_test_policy_divergence.pkl', 'rb') as f:
    #     record = pickle.load(f)
    # plt.figure()
    # kl = [x[0] for x in record]
    # ratio = [x[2] / x[1] for x in record]
    # plt.scatter(kl, ratio)
    # plt.savefig('figures/ST_MT_test_policy_divergence.png')
    # plt.close()
    
    # context_features = [
    #     "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb model
    #     "body_mass", "body_shape", # limb hardware
    #     "jnt_pos", # joint model
    #     "joint_range", "joint_axis", "gear" # joint hardware
    # ]

    # for feature in context_features:
    #     plot_context_hist(feature)
    
    # folders = [
    #     'log_HN_fix_normalization_wo_PE/1409', 
    #     'log_baseline_wo_PE/1410', 
    #     'log_origin', 
    # ]
    # plot_grad_norm(folders, key='approx_kl')

    # results = [
    #     'log_HN_fix_normalization_wo_PE_1409', 
    #     'log_HN_fix_normalization_wo_PE_1410', 
    #     'log_baseline_wo_PE_norm_twice_1409', 
    #     'log_baseline_wo_PE_norm_twice_1410', 
    #     'log_baseline_1409', 
    #     'log_baseline_1410', 
    #     'log_HN_fix_normalization_PE_in_base_1409', 
    #     'log_HN_fix_normalization_PE_in_base_1410', 
    # ]
    # plot_test_performance(results)

    # folders = [
    #     'hist_ratio_baseline', 
    #     'hist_ratio_wo_dropout', 
    # ]
    # folders = [
    #     'ST_100M_floor-5506-11-8-01-12-33-50', 
    #     'ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50', 
    # ]
    # agents = os.listdir('output/log_single_task_wo_dropout')
    # plot_training_stats(folders, key='ratio', suffix=None)

    # folders = [
    #     'csr_baseline', 
    #     'csr_baseline_wo_dropout', 
    #     # 'csr_fixed_attention_wo_PE+dropout', 
    #     # 'csr_fix_attention_MLP_wo_PE+dropout', 
    #     'csr_baseline_KL_5_wo_PE+dropout', 
    #     'csr_fix_attention_MLP_KL_5_wo_PE+dropout', 
    # ]
    # names = [
    #     'baseline + dropout', 
    #     'baseline', 
    #     'baseline + KL=0.05', 
    #     'baseline + KL=0.05 + fix attention', 
    # ]
    # kl = [0.2, 0.2, 0.05, 0.05]
    # prefix = 'csr'
    # folders = [
    #     'ft_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_HN_fix_attention_MLP_KL_5_wo_PE+dropout', 
    #     'ft_HN_fix_attention_MLP_wo_PE+dropout', 
    #     'ft_baseline_KL_5_wo_PE+dropout', 
    #     'ft_baseline_dropout_wo_PE', 
    # ]
    # names = [
    #     'baseline', 
    #     'baseline wo dropout', 
    #     'fixed attention, dropout, wo PE', 
    #     'fixed attention, dropout both, wo PE', 
    #     'KL threshold = 0.05, wo dropout', 
    #     'fixed attention, KL threshold = 0.05, wo dropout', 
    # ]
    # names = folders
    # kl = [0.05, 0.05, 0.2, 0.05, 0.2]
    # prefix = 'ft'

    # folders = [
    #     'incline_baseline', 
    #     'incline_baseline_KL_5_wo_dropout', 
    #     'incline_MLP_fix_attention_KL_5_wo_dropout', 
    #     'incline_baseline_wo_dropout', 
    #     'incline_MLP_fix_attention_wo_dropout', 
    # ]
    # names = folders
    # kl = [0.2, 0.05, 0.05, 0.2, 0.2]
    # prefix = 'incline'

    # folders = [
    #     'exploration_baseline', 
    #     'exploration_baseline_KL_5_wo_dropout', 
    #     # 'exploration_baseline_wo_dropout', 
    #     'exploration_MLP_fix_attention_KL_5_wo_dropout', 
    #     # 'exploration_MLP_fix_attention_wo_dropout', 
    # ]
    # names = folders
    # kl = [0.2, 0.05, 0.05]
    # prefix = 'exploration'

    # for stat in ['approx_kl', 'grad_norm', 'pi_loss', 'ratio', 'val_loss', 'clip_frac']:
    # # for stat in ['clip_frac']:
    #     plot_training_stats(folders, names, key=stat, prefix=prefix, kl_threshold=kl)
        # agents = [
        #     'floor-1409-11-14-01-15-44-14', 
        #     'floor-5506-11-8-01-12-33-50', 
        #     'mvt-5506-12-4-17-12-01-27', 
        # ]
        # for agent in agents:
        #     # folders = [
        #     #     f'log_single_task_wo_dropout/{agent}', 
        #     #     f'log_single_task/{agent}', 
        #     #     f'log_single_task_wo_pe+dropout/{agent}', 
        #     # ]
        #     folders = [
        #         f'ST_100M_{agent}', 
        #         f'ST_100M_wo_dropout_{agent}', 
        #     ]
        #     names = [
        #         'ST baseline', 
        #         'ST wo dropout', 
        #     ]
        #     plot_training_stats(folders, names=names, key=stat, prefix=f'ST_100M_{agent}')
            # analyze_ratio_hist_trend(folders, prefix=agent)
    # scatter_train_performance(folders)
    # scatter_train_performance_compared_to_subset_train('log_baseline', 'ST-subset')
    # get_context()

    folders = [
        'hist_ratio_baseline', 
        'hist_ratio_wo_dropout', 
    ]
    # folders = [folders[0], folders[1]]
    # analyze_ratio_hist_trend(folders)
    # test_init_ratio_hist(folders)

    score_files = [
        'log_baseline_1409_terminate_on_fall', 
        'log_baseline_1409_deterministic', 
        'log_baseline_1409_terminate_on_fall_deterministic', 
        'log_baseline_1409', 
    ]
    fig_name = 'compare_mode_baseline_1409'

    score_files = [
        'log_baseline_1410_terminate_on_fall', 
        'log_baseline_1410_deterministic', 
        'log_baseline_1410_terminate_on_fall_deterministic', 
        'log_baseline_1410', 
    ]
    fig_name = 'compare_mode_baseline_1410'

    score_files = [
        'log_fix_attention_wo_PE_1409_terminate_on_fall', 
        'log_fix_attention_wo_PE_1409_deterministic', 
        'log_fix_attention_wo_PE_1409_terminate_on_fall_deterministic', 
        'log_fix_attention_wo_PE_1409', 
    ]
    fig_name = 'compare_mode_fix_attention_wo_PE_1409'

    # analyze_ratio_hist_per_epoch('ST_100M_wo_dropout_floor-5506-11-8-01-12-33-50')

    # score_files = [
    #     'log_HN_fix_attention_1409_terminate_on_fall', 
    #     'log_HN_fix_attention_1409_deterministic', 
    #     'log_HN_fix_attention_1409_terminate_on_fall_deterministic', 
    #     'log_HN_fix_attention_1409', 
    # ]
    # fig_name = 'compare_mode_HN_fix_attention_1409'

    # plot_test_score(score_files, fig_name)
    # scores = []
    # for x in score_files:
    #     with open(f'output/eval/{x}.pkl', 'rb') as f:
    #         scores.append(pickle.load(f))
    # scatter_test_score(scores[0], scores[1], score_files[0], score_files[1])
    # scatter_test_score('log_baseline_wo_PE+dropout_1409', 'log_baseline_wo_PE+dropout_1410')
    # scatter_train_score('log_fix_attention_wo_PE/1409', 'log_fix_attention_wo_PE/1410')