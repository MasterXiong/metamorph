import argparse
import os

from tools.evaluate import evaluate_model


def evaluate_checkpoint(folder, test_set, interval=600):

    iteration = interval
    while (1):
        test_set_name = test_set.split('/')[1]
        suffix = f'{test_set_name}_cp_{iteration}'
        model_path = f'{folder}/checkpoint_{iteration}.pt'
        agent_path = test_set
        policy_folder = folder
        print (model_path)
        folder_name = folder.split('/')[1]
        seed = folder.split('/')[2]
        if os.path.exists(f'eval/{folder_name}/{folder_name}_{seed}_{suffix}.pkl'):
            iteration += interval
            continue
        if not os.path.exists(model_path):
            break
        evaluate_model(model_path, agent_path, policy_folder, suffix=suffix)
        iteration += interval



if __name__ == '__main__':
    
    # python tools/eval_learning_curve.py --folder output/ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout/1409 --interval 100
    # python tools/eval_learning_curve.py --folder output/ft_debug_1000M_mutate_1000_PVL_KL_5_wo_PE+dropout/1409
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--folder", type=str)
    parser.add_argument("--test_set", default='unimals_100/test', type=str)
    parser.add_argument("--interval", default=600, type=int)
    args = parser.parse_args()

    evaluate_checkpoint(args.folder, args.test_set, args.interval)