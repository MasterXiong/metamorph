import argparse
import os

from tools.evaluate import evaluate_model


def evaluate_checkpoint(folder, test_set, interval=600, additional_suffix=None):

    iteration = interval
    while (1):
        test_set_name = test_set.split('/')[1]
        seed = folder.split('/')[-1]
        # suffix = f'{test_set_name}_cp_{iteration}_wo_height_check'
        suffix = f'{seed}_{test_set_name}_cp_{iteration}'
        if additional_suffix is not None:
            suffix = suffix + '_' + additional_suffix
        model_path = f'{folder}/checkpoint_{iteration}.pt'
        agent_path = test_set
        policy_folder = folder
        print (model_path)
        folder_name = folder.split('/')[1]
        seed = folder.split('/')[2]
        if os.path.exists(f'eval/{folder_name}/{suffix}.pkl'):
            iteration += interval
            continue
        if not os.path.exists(model_path):
            break
        evaluate_model(model_path, agent_path, policy_folder, suffix=suffix, compute_gae=False, \
            terminate_on_fall=True)
        iteration += interval



if __name__ == '__main__':
    
    # python tools/eval_learning_curve.py --folder output/ft_400M_init_base_generation_LP_curation_uniform_KL_5_wo_PE+dropout/1409 --test_set data/test --interval 100 --suffix wo_reward_update
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