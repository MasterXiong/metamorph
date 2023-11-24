import argparse
import os
import numpy as np

from tools.evaluate import evaluate_model


def evaluate_checkpoint(folder, test_set, interval=600, additional_suffix=None, seeds=None):

    if seeds is None:
        seeds = os.listdir(folder)
    all_seed_scores = []
    for seed in seeds:
        seed_scores = {}
        iteration = interval
        while (1):
            test_set_name = test_set.split('/')[1]
            # suffix = f'{test_set_name}_cp_{iteration}_wo_height_check'
            suffix = f'{seed}_{test_set_name}_cp_{iteration}'
            if additional_suffix is not None:
                suffix = suffix + '_' + additional_suffix
            model_path = f'{folder}/{seed}/checkpoint_{iteration}.pt'
            agent_path = test_set
            policy_folder = f'{folder}/{seed}'
            print (model_path)
            # if os.path.exists(f'eval/{folder_name}/{suffix}.pkl'):
            #     iteration += interval
            #     continue
            if not os.path.exists(model_path):
                break
            score = evaluate_model(model_path, agent_path, policy_folder, suffix=suffix, compute_gae=False, \
                terminate_on_fall=True)
            seed_scores[iteration] = score
            iteration += interval
        all_seed_scores.append(seed_scores)
    for iteration in all_seed_scores[0].keys():
        avg_score = np.mean([seed_scores[iteration] for seed_scores in all_seed_scores])
        print ([seed_scores[iteration] for seed_scores in all_seed_scores])
        std = np.std([seed_scores[iteration] for seed_scores in all_seed_scores])
        print (f'iteration {iteration}: {avg_score} +- {std} ({len(all_seed_scores)} seeds)')



if __name__ == '__main__':
    
    # python tools/eval_learning_curve.py --folder output/ft_MLP_HN_IO_MLP_encoder_bias_init_maximin_sample_KL_5_wo_PE+dropout --test_set data/test --interval 1200 --suffix wo_reward_update
    # python tools/eval_learning_curve.py --folder output/ft_400M_mutate_400_env_256_uniform_sample_KL_5_wo_PE+dropout/1409 --test_set unimals_100/train --interval 100
    # python tools/eval_learning_curve.py --folder output/ft_400M_baseline_uniform_sample_KL_5_wo_PE+dropout/1409 --test_set unimals_100/train_remove_level_1 --interval 100
    # python tools/eval_learning_curve.py --folder output/ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout/1409
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--test_set", default='unimals_100/test', type=str)
    parser.add_argument("--interval", default=600, type=int)
    parser.add_argument("--suffix", type=str)
    args = parser.parse_args()

    evaluate_checkpoint(args.folder, args.test_set, args.interval, args.suffix)