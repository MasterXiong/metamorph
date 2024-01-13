import argparse
import os
import sys

from metamorph.algos.ppo.distill import *
from metamorph.config import cfg
from metamorph.config import dump_cfg

from tools.train_ppo import set_cfg_options


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "--no_context_in_state", action="store_true"
    )
    parser.add_argument(
        "--validation", action="store_true"
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



if __name__ == '__main__':

    # MT distill to MLP
    # python tools/distill.py --cfg configs/ft.yaml RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1410 OUT_DIR distilled_policy/MT_TF_to_MLP_lr_1e-3/1409 MODEL.TYPE vanila_mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 20
    # MT distill to PerAgentMLP
    # python tools/distill.py --cfg configs/ft.yaml RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 OUT_DIR distilled_policy/MT_TF_to_PAMLP_lr_1e-3/1409 MODEL.TYPE per_agent_mlp MODEL.MLP.LAYRT_NUM 2 DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 100 DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    # python tools/distill.py --cfg configs/csr.yaml RNG_SEED 1409 DISTILL.SOURCE csr_200M_baseline_KL_3_wo_PE+dropout/1409 OUT_DIR distilled_policy/csr_MT_TF_to_PAMLP_lr_1e-3_batchsize_10240/1409 MODEL.TYPE per_agent_mlp MODEL.MLP.LAYER_NUM 2 DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 100 DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True DISTILL.BATCH_SIZE 10240
    # MT distill to HN-MLP
    # python tools/distill.py --cfg configs/ft.yaml --no_context_in_state RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 MODEL.TYPE mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 MODEL.MLP.HN_INPUT True MODEL.MLP.HN_OUTPUT True MODEL.MLP.HN_HIDDEN True MODEL.MLP.LAYER_NUM 2 MODEL.MLP.INPUT_AGGREGATION limb_num MODEL.MLP.HN_GENERATE_BIAS True MODEL.MLP.CONTEXT_MASK True MODEL.MLP.CONTEXT_ENCODER_TYPE transformer MODEL.MLP.HN_INIT_STRATEGY bias_init OUT_DIR distilled_policy/MT_TF_to_HN-MLP_lr_1e-3_decouple_grad_norm_0.5_KL_loss_balanced/1409 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 50 DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    # python tools/distill.py --cfg configs/csr.yaml --no_context_in_state RNG_SEED 1409 DISTILL.SOURCE csr_200M_baseline_KL_3_wo_PE+dropout/1409 MODEL.TYPE mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 MODEL.MLP.HN_INPUT True MODEL.MLP.HN_OUTPUT True MODEL.MLP.HN_HIDDEN True MODEL.MLP.LAYER_NUM 2 MODEL.MLP.INPUT_AGGREGATION limb_num MODEL.MLP.HN_GENERATE_BIAS True MODEL.MLP.CONTEXT_MASK True MODEL.MLP.CONTEXT_ENCODER_TYPE transformer MODEL.MLP.HN_INIT_STRATEGY bias_init OUT_DIR distilled_policy/csr_MT_TF_to_HN-MLP_v3_lr_1e-3_decouple_grad_norm_0.5_KL_loss_balanced/1409 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 50 DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    # python tools/distill.py --cfg configs/obstacle.yaml --no_context_in_state RNG_SEED 1409 DISTILL.SOURCE obstacle_200M_baseline_KL_3_wo_PE+dropout/1409 MODEL.TYPE mlp DISTILL.PER_AGENT_SAMPLE_NUM 16000 MODEL.MLP.HN_INPUT True MODEL.MLP.HN_OUTPUT True MODEL.MLP.HN_HIDDEN True MODEL.MLP.LAYER_NUM 2 MODEL.MLP.INPUT_AGGREGATION limb_num MODEL.MLP.HN_GENERATE_BIAS True MODEL.MLP.CONTEXT_MASK True MODEL.MLP.CONTEXT_ENCODER_TYPE transformer MODEL.MLP.HN_INIT_STRATEGY bias_init OUT_DIR distilled_policy/obstacle_MT_TF_to_HN-MLP_v3_lr_1e-3_decouple_grad_norm_0.5_dropout_KL_loss_balanced/1409 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 200 DISTILL.GRAD_NORM 0.5 MODEL.MLP.CONTEXT_EMBEDDING_DROPOUT True DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    # MT distill to new version of HN-MLP
    # python tools/distill.py --cfg configs/ft.yaml RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 MODEL.TYPE hnmlp MODEL.MLP.LAYER_NUM 2 OUT_DIR distilled_policy/test/1409 DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 100 DISTILL.VALUE_NET False
    # python tools/distill.py --cfg configs/ft.yaml RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 MODEL.TYPE hnmlp MODEL.MLP.LAYER_NUM 2 MODEL.HYPERNET.HN_INIT_STRATEGY bias_init_v3 OUT_DIR distilled_policy/MT_TF_to_HN-MLP_v2_bias_init_v3_lr_1e-4/1409 DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.0001 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 100
    # python tools/distill.py --cfg configs/ft.yaml --no_context_in_state RNG_SEED 1409 DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 MODEL.TYPE hnmlp MODEL.MLP.LAYER_NUM 2 MODEL.HYPERNET.HN_INIT_STRATEGY bias_init_v2 OUT_DIR distilled_policy/MT_TF_to_HN-MLP_v2_decouple_bias_init_v2_dropout_0.1_act_mean_lr_1e-3/1409 DISTILL.PER_AGENT_SAMPLE_NUM 16000 DISTILL.BASE_LR 0.001 DISTILL.SAVE_FREQ 5 DISTILL.EPOCH_NUM 50 MODEL.HYPERNET.EMBEDDING_DROPOUT 0.1
    # ST distill to HN-MLP
    # python tools/distill.py --cfg configs/csr.yaml --no_context_in_state RNG_SEED 1409 DISTILL.SOURCE MLP_ST_csr_256*2_KL_3/1409 MODEL.TYPE mlp DISTILL.PER_AGENT_SAMPLE_NUM 64000 MODEL.MLP.HN_INPUT True MODEL.MLP.HN_OUTPUT True MODEL.MLP.HN_HIDDEN True MODEL.MLP.LAYER_NUM 2 MODEL.MLP.INPUT_AGGREGATION limb_num MODEL.MLP.HN_GENERATE_BIAS True MODEL.MLP.CONTEXT_MASK True MODEL.MLP.CONTEXT_ENCODER_TYPE transformer MODEL.MLP.HN_INIT_STRATEGY bias_init OUT_DIR distilled_policy/csr_ST_MLP_to_HN-MLP_v3_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_64000/1409 DISTILL.PER_AGENT_SAMPLE_NUM 64000 DISTILL.BASE_LR 0.0003 DISTILL.SAVE_FREQ 5 DISTILL.EPOCH_NUM 50 DISTILL.GRAD_NORM 0.5 MODEL.MLP.CONTEXT_EMBEDDING_DROPOUT True DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    # ST distill to TF
    # python tools/distill.py --cfg configs/csr.yaml RNG_SEED 1409 DISTILL.SOURCE MLP_ST_csr_256*2_KL_3/1409 MODEL.TYPE transformer MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None OUT_DIR distilled_policy/csr_ST_MLP_to_TF_lr_3e-4_expert_size_64000/1409 DISTILL.PER_AGENT_SAMPLE_NUM 64000 DISTILL.BASE_LR 0.0003 DISTILL.SAVE_FREQ 5 DISTILL.EPOCH_NUM 50 DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    if args.no_context_in_state:
        obs_type = [
            "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", # limb
            "qpos", "qvel", # joint
        ]
        ob_opts = ["MODEL.PROPRIOCEPTIVE_OBS_TYPES", obs_type]
        cfg.merge_from_list(ob_opts)
    cfg.ENV.WALKER_DIR = 'data/train'
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dump_cfg()

    agent_path = 'data/train'
    if 'ST' in cfg.OUT_DIR:
        teacher_mode = 'ST'
    else:
        teacher_mode = 'MT'
    print (teacher_mode)
    distill_policy(cfg.DISTILL.SOURCE, cfg.DISTILL.TARGET, agent_path, teacher_mode, validation=args.validation)
