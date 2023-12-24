import os

from metamorph.algos.ppo.distill import *
from metamorph.config import cfg
from metamorph.config import dump_cfg

from tools.train_ppo import parse_args, set_cfg_options



if __name__ == '__main__':

    model_path = 'baselines/ft_baseline_KL_5_wo_PE+dropout/1409'
    agent_path = 'data/train'
    agents = [x[:-4] for x in os.listdir(f'{agent_path}/xml')]
    # generate_expert_data(model_path, agent_path)

    # python tools/distill.py --cfg configs/ft.yaml DISTILL.SOURCE ft_baseline_KL_5_wo_PE+dropout/1409 DISTILL.TARGET MT_TF_to_MLP/1409 MODEL.TYPE vanila_mlp
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.ENV.WALKER_DIR = 'data/train'
    set_cfg_options()
    dump_cfg()

    distill_policy(cfg.DISTILL.SOURCE, cfg.DISTILL.TARGET, agents)
