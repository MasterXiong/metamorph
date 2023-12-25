import argparse
import os

from metamorph.algos.ppo.distill import *
from metamorph.config import cfg
from metamorph.config import dump_cfg

from tools.train_ppo import parse_args, set_cfg_options



if __name__ == '__main__':

    # python tools/distill.py --model_path baselines/ft_baseline_KL_5_wo_PE+dropout/1411 --start 0 --end 25
    # parser = argparse.ArgumentParser(description="Collect expert data from trained RL agent")
    # parser.add_argument("--model_path", help="the path of the expert", required=True, type=str)
    # parser.add_argument("--start", help="the start index", type=int, default=0)
    # parser.add_argument("--end", help="the end index", type=int, default=100)
    # args = parser.parse_args()

    # model_path = args.model_path
    # agent_path = 'data/train'
    # generate_expert_data(model_path, agent_path, start=args.start, end=args.end)

    # MT distill to MLP
    # python tools/distill.py --cfg configs/ft.yaml DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1410 OUT_DIR distilled_policy/MT_TF_to_MLP_lr_1e-3/1409 MODEL.TYPE vanila_mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 20
    # MT distill to HN-MLP
    # python tools/distill.py --cfg configs/ft.yaml DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1410 MODEL.TYPE mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 MODEL.MLP.HN_INPUT True MODEL.MLP.HN_OUTPUT True MODEL.MLP.HN_HIDDEN True MODEL.MLP.LAYER_NUM 2 MODEL.MLP.INPUT_AGGREGATION limb_num MODEL.MLP.HN_GENERATE_BIAS True MODEL.MLP.CONTEXT_MASK True MODEL.MLP.CONTEXT_ENCODER_TYPE transformer MODEL.MLP.HN_INIT_STRATEGY bias_init OUT_DIR distilled_policy/MT_TF_to_HN-MLP_lr_1e-3/1409 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 20 DISTILL.BATCH_SIZE 5120

    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.ENV.WALKER_DIR = 'data/train'
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dump_cfg()

    agent_path = 'data/train'
    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')]
    distill_policy(cfg.DISTILL.SOURCE, cfg.DISTILL.TARGET, agents)
