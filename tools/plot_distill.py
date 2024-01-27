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


def distill_evaluation_curve(folders, names, test_set, suffix, plot_each_seed=False, bound=None):

    colors = plt.get_cmap('tab10').colors

    plt.figure()
    count = 0
    max_epoch = 0
    for i, folder in enumerate(folders):
        print (folder)
        try:
            seeds = os.listdir(f'distilled_policy/{folder}')
        except:
            seeds = os.listdir(f'dagger_policy/{folder}')
        all_seed_x, all_seed_y = [], []
        for seed in seeds:
            xdata, ydata = [], []
            iteration = 0
            while (1):
                file_path = f'eval/{folder}/{seed}_{test_set}_cp_{iteration}_deterministic.pkl'
                if not os.path.exists(file_path):
                    if iteration == 0:
                        xdata.append(0)
                        ydata.append(0.)
                        iteration += folders[folder]
                        continue
                    else:
                        break
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                avg_score = np.mean([np.array(results[agent][0]).mean() for agent in results])
                print (iteration, len(results.keys()))
                xdata.append(iteration)
                ydata.append(avg_score)
                iteration += folders[folder]
            if iteration != folders[folder]:
                all_seed_x.append(xdata)
                all_seed_y.append(ydata)
        seed_num = len(all_seed_x)
        if seed_num != 0:
            print (all_seed_y)
            min_len = min([len(x) for x in all_seed_x])
            plot_x = all_seed_x[0][:min_len]
            plot_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).mean(axis=0)
            error_y = np.stack([np.array(x[:min_len]) for x in all_seed_y]).std(axis=0)
            # print (plot_x, plot_y)
            print (f'final score: {plot_y[-1]} +- {error_y[-1]}')
            plt.errorbar(plot_x, plot_y, yerr=error_y, c=colors[i], label=f'{names[i]} (deterministic) ({seed_num} seeds)')
        max_epoch = max(max_epoch, plot_x[-1])
    if bound is not None:
        bound_mean, bound_std = np.mean(bound), np.std(bound)
        plt.plot([plot_x[0], max_epoch], [bound_mean, bound_mean], 'k--')
        plt.fill_between([plot_x[0], max_epoch], bound_mean - bound_std, bound_mean + bound_std, color='k', alpha=0.25)
    plt.legend(loc='lower right', ncols=1, prop = {'size':5})
    plt.title(f'Generalization to {test_set}')
    plt.xlabel('Distill epoch')
    plt.ylabel('Return')
    plt.savefig(f'figures/distill_evaluation_{test_set}_{suffix}.png')
    plt.close()


def plot_loss_curve(folders, names, suffix):
    colors = plt.get_cmap('tab10').colors
    plt.figure()
    for i, folder in enumerate(folders):
        try:
            curves = []
            for seed in os.listdir(f'distilled_policy/{folder}'):
                with open(f'distilled_policy/{folder}/{seed}/loss_curve.pkl', 'rb') as f:
                    curve = pickle.load(f)
                    if len(curve) == 3:
                        curves.append(curve[0])
                    else:
                        curves.append(curve)
            iter_num = min([len(x) for x in curves])
            curves = np.stack([x[:iter_num] for x in curves], axis=0)
            mean = curves.mean(0)
            std = curves.std(0)
            plt.plot(mean, label=names[i], c=colors[i])
            plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.25)
        except:
            continue
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'figures/distill_loss_curve_{suffix}.png')
    plt.close()


def plot_dagger_loss_curve(folders, names, suffix):
    colors = plt.get_cmap('tab10').colors
    plt.figure()
    for i, folder in enumerate(folders):
        train_curves, test_curves = [], []
        for seed in os.listdir(f'dagger_policy/{folder}'):
            if not os.path.exists(f'dagger_policy/{folder}/{seed}/loss_curves.pkl'):
                continue
            with open(f'dagger_policy/{folder}/{seed}/loss_curves.pkl', 'rb') as f:
                curve = pickle.load(f)
            train_curves.append(curve[0])
            test_curves.append(curve[1])
        if len(train_curves) == 0:
            continue
        iter_num = min([len(x) for x in train_curves])
        curves = np.stack([x[:iter_num] for x in train_curves], axis=0)
        mean = curves.mean(0)
        std = curves.std(0)
        plt.plot(mean, label=names[i], color=colors[i])
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, color=colors[i], alpha=0.25)
        iter_num = min([len(x) for x in test_curves])
        curves = np.stack([x[:iter_num] for x in test_curves], axis=0)
        mean = curves.mean(0)
        std = curves.std(0)
        plt.plot(mean, color=colors[i], linestyle='--')
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, color=colors[i], alpha=0.25)
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'figures/dagger_loss_curve_{suffix}.png')
    plt.close()



bound = {
    'train': [2630, 2654, 2450], 
    'test': [1173, 1123, 1173], 
}
folders = {
    # 'obstacle_MT_modumorph_to_TF_lr_3e-4_expert_size_64000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
    # 'obstacle_MT_modumorph_to_modumorph_lr_3e-4_expert_size_64000': 10, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 5, 
    'obstacle_MT_modumorph_to_modumorph_lr_3e-4_expert_size_8k*1000': 10, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 30, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_10_anneal_1_act_sample_iter_reset': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_base_LN_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_paper_init_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_heuristic_init_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_L2_reg_1e-5': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_L2_reg_1e-5_adamw': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_context_embed_init_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_conttext_v3_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_skip_connection_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 5, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_HN_hfield_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_dagger_cp_10': 5, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_dagger_cp_0_full_anneal': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_dagger_cp_0': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_context_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 30, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss_th_2': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_mean_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_mean_expert_size_8k*1000_weighted_loss': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8ep*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_16ep*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_10_anneal_1_act_mean': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_10_anneal_1_act_sample': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_none_dropout_KL_loss_balanced_expert_size_8k*1000': 30, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_2_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_4_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_1_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_gnn_CE_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*4_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 30, 
}
names = [
    # 'TF', 
    # 'HN-MLP (256*3)', 
    'HN-MLP, hfield hidden', 
    # 'modumorph', 
    'HN-MLP, hfield hidden, 8k*1000', 
    'modumorph, 8k*1000', 
    'HN-MLP, hfield hidden, 8k*1000, sum agg', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, cp 10, anneal 1, act sample, iter reset', 
    # 'HN-MLP, sum agg, base LN', 
    # 'HN-MLP, sum agg, paper init', 
    # 'HN-MLP, sum agg, heuristic init', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, L2 reg', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, L2 reg, AdamW', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, context embed init', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, context v3', 
    # 'HN-MLP, hfield hidden, 8k*1000, skip connection', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, HN hfield', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, cp 10', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, full anneal', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, anneal 10', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, context sum agg', 
    'HN-MLP, hfield hidden, 8k*1000, sum agg, weighted action loss', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, weighted action loss (th=2)', 
    # 'HN-MLP, hfield hidden, 8k*1000, mean agg, weighted action loss', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, weighted action loss, mean distillation loss', 
    # 'HN-MLP, hfield hidden, 8ep*1000, sum agg', 
    # 'HN-MLP, hfield hidden, 16ep*1000, sum agg', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, cp 10, anneal 1, act mean', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, dagger, cp 10, anneal 1, act sample', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, norm=none', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, norm=2', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, norm=4', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, norm=1', 
    # 'HN-MLP, hfield hidden, 8k*1000, sum agg, GNN context encoder', 
    'HN-MLP (256*4), hfield hidden, 8k*1000, sum agg', 
]
distill_evaluation_curve(folders, names, 'test', 'obstacle_MM', bound=bound['test'])
distill_evaluation_curve(folders, names, 'train', 'obstacle_MM', bound=bound['train'])
plot_loss_curve(folders, names, 'obstacle_MM')

folders = {
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_0_anneal_1': 40, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_0_anneal_1_default_100': 40, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_0_anneal_1_with_dropout': 40, 
    'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_cp_10_anneal_1_with_dropout': 10, 
}
names = [
    'HN-MLP, cp 0, anneal 1', 
    'HN-MLP, cp 0, anneal 1, train on default 100', 
    'HN-MLP, cp 0, anneal 1, with dropout', 
    'HN-MLP, cp 10, anneal 1, with dropout', 
]
distill_evaluation_curve(folders, names, 'train', 'obstacle_dagger', bound=bound['train'])
distill_evaluation_curve(folders, names, 'test', 'obstacle_dagger', bound=bound['test'])
plot_dagger_loss_curve(folders, names, 'obstacle')

bound = {
    'train': [2225, 2257, 2012], 
    'test': [1025, 874, 711], 
}
folders = {
    # 'csr_MT_modumorph_to_TF_lr_3e-4_expert_size_64000': 10, 
    # 'csr_MT_modumorph_to_HN-MLP_v3_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
    # 'csr_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
    # 'csr_MT_modumorph_to_modumorph_lr_3e-4_expert_size_64000': 10, 
    'csr_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    'csr_MT_modumorph_to_modumorph_lr_3e-4_expert_size_8k*1000': 10, 
    'csr_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    'csr_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss': 10, 
    'csr_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss_th_2': 10, 
    'csr_MT_modumorph_to_modumorph_compressed_lr_3e-4_expert_size_8k*1000': 5, 
}
names = [
    # 'TF', 
    # 'HN-MLP (256*3)', 
    # 'HN-MLP, hfield hidden', 
    # 'modumorph', 
    'HN-MLP, hfield hidden, 8k*1000', 
    'modumorph, 8k*1000', 
    'HN-MLP, hfield hidden, 8k*1000, sum agg', 
    'HN-MLP, hfield hidden, 8k*1000, sum agg, weighted loss', 
    'HN-MLP, hfield hidden, 8k*1000, sum agg, weighted loss (th=2)', 
    'modumorph (compressed), 8k*1000', 
]
distill_evaluation_curve(folders, names, 'test', 'csr_MM', bound=bound['test'])
distill_evaluation_curve(folders, names, 'train', 'csr_MM', bound=bound['train'])
plot_loss_curve(folders, names, 'csr_MM')

# folders = {
#     'ft_MT_TF_to_TF_lr_3e-4_expert_size_64000': 10, 
#     'ft_MT_TF_to_HN-MLP_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
#     'ft_ST_MLP_to_HN-MLP_v3_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000': 10, 
#     'ft_ST_MLP_to_TF_lr_3e-4_expert_size_64000': 10, 
# }
# names = [
#     'TF', 
#     'HN-MLP', 
#     'HN-MLP, ST teachers', 
#     'TF, ST teachers', 
# ]
# distill_evaluation_curve(folders, names, 'test', 'ft_ST', bound=test_bound['ft'])
# distill_evaluation_curve(folders, names, 'train', 'ft_ST', bound=train_bound['ft'])
# plot_loss_curve(folders, names, 'ft_ST')

bound = {
    'train': [4294, 4405, 4377], 
    'test': [1302, 1303, 1294], 
}
folders = {
    'ft_MT_modumorph_to_HN-MLP_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    'ft_MT_modumorph_to_modumorph_lr_3e-4_expert_size_8k*1000': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_skip_connection_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 30, 
    'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_weighted_loss': 30, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_KL_loss_balanced_expert_size_8k*1000_wo_dropout': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_base_LN_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_paper_init_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_L2_reg_1e-5': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_KL_loss_balanced_expert_size_8k*1000_L2_reg_1e-5_wo_dropout': 10, 
    # 'ft_MT_modumorph_to_HN-MLP_sum_agg_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000_dagger_cp_10': 10, 
    'ft_MT_modumorph_to_modumorph_compressed_lr_3e-4_expert_size_8k*1000': 5, 
}
names = [
    'HN-MLP', 
    'modumorph', 
    # 'HN-MLP, skip connection', 
    'HN-MLP, sum agg', 
    'HN-MLP, sum agg, weighted loss', 
    # 'HN-MLP, sum agg, wo dropout', 
    # 'HN-MLP, sum agg, base LN', 
    # 'HN-MLP, sum agg, paper init', 
    # 'HN-MLP, L2 reg', 
    # 'HN-MLP, L2 reg, wo dropout', 
    # 'HN-MLP, sum_agg, dagger', 
    'modumorph (compressed), 8k*1000', 
]
distill_evaluation_curve(folders, names, 'test', 'ft_MM', bound=bound['test'])
distill_evaluation_curve(folders, names, 'train', 'ft_MM', bound=bound['train'])
plot_loss_curve(folders, names, 'ft_MM')


# def scatter_test_score(folder_x, folder_y):
#     agents = [x[:-4] for x in os.listdir('data/test/xml')]
#     plt.figure()

#     def compute_avg_score(folder):
#         avg_score = []
#         for seed in [1409, 1410, 1411]:
#             with open(f'eval/{folder}/{seed}_test_cp_20_deterministic.pkl', 'rb') as f:
#                 score = pickle.load(f)
#             avg_score.append(np.array([np.mean(score[agent][0]) for agent in agents]))
#         avg_score = np.stack(avg_score).mean(0)
#         return avg_score

#     score_x = compute_avg_score(folder_x)
#     score_y = compute_avg_score(folder_y)
#     plt.scatter(score_x, score_y)
#     print (np.mean(score_x))
#     print (np.mean(score_y))
#     plt.plot([500, 5500], [500, 5500], 'k-')
#     plt.xlabel(folder_x)
#     plt.ylabel(folder_y)
#     os.makedirs('figures/compare_test_score', exist_ok=True)
#     plt.savefig(f'figures/compare_test_score/compare.png')
#     plt.close()

# folder_x = 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000'
# folder_y = 'obstacle_MT_modumorph_to_HN-MLP_v3_256*3_hfield_hidden_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000'
# scatter_test_score(folder_x, folder_y)
