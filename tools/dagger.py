import os

from metamorph.config import cfg
from metamorph.algos.distill.dagger import DAggerTrainer

from tools.train_ppo import parse_args, set_cfg_options, dump_cfg


def dagger_train():
    dagger_trainer = DAggerTrainer()
    dump_cfg()
    dagger_trainer.train()


def main():
    # TODO: set the output dir name
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    # Set cfg options which are inferred
    set_cfg_options()

    cfg.OUT_DIR = f'{cfg.DAGGER.STUDENT_PATH[:-5]}_{cfg.DAGGER.SUFFIX}/{cfg.DAGGER.STUDENT_PATH[-4:]}'
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dagger_train()



if __name__ == "__main__":
    # python tools/dagger.py --cfg configs/ft.yaml ENV.WALKER_DIR data/train_mutate_1000 DAGGER.TEACHER_PATH baselines/ft_HN+FA_KL_5_wo_PE+dropout/1409 DAGGER.STUDENT_PATH distilled_policy/ft_MT_modumorph_to_HN-MLP_lr_3e-4_decouple_grad_norm_0.5_dropout_KL_loss_balanced_expert_size_8k*1000
    main()