# docker cmd
git clone https://github.com/oxwhirl/luisa_docker.git
bash build.sh Dockerfile_mj150

docker run --gpus device=0 --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph:/user/metamorph -it metamorph /bin/bash
NV_GPU=1 nvidia-docker run --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph:/user/metamorph -it metamorph /bin/bash
NV_GPU=1 nvidia-docker run --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph:/user/metamorph -it metamorphv2 /bin/bash
docker run --gpus device=0 --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph:/user/metamorph -v /data/tucana/zheong/output/:/user/metamorph/output -it metamorphv2 /bin/bash
NV_GPU=0 nvidia-docker run --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph:/user/metamorph -v /data/dgx1/zheong/metamorph/:/user/metamorph/output -it metamorphv2 /bin/bash

NV_GPU=0 nvidia-docker run --rm --network host --ipc=host --user $(id -u) -v /home/zheong/metamorph_ST:/user/metamorph -v /data/tucana/zheong/metamorph:/user/metamorph/output -it metamorph /bin/bash

docker run --gpus device=0 --rm --network host --ipc=host --user $(id -u) -v /home/zheong/morphology-design:/user/metamorph -it morphology_design /bin/bash

# just for test
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR test ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.BASE_CONTEXT_NORM fixed

# PE project
python tools/train_ppo.py --cfg ./configs/incline.yaml OUT_DIR ./output/incline_decoder_64_separate_PE_zero_init_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.DECODER_DIMS [64] MODEL.TRANSFORMER.USE_SEPARATE_PE True

python tools/train_ppo.py --cfg ./configs/incline.yaml OUT_DIR ./output/incline_decoder_64_PE_zero_init_KL_3_wo_dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.DECODER_DIMS [64]

# MT train with constant lr
# ft
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_constant_lr_0.0002_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. PPO.LR_POLICY constant PPO.BASE_LR 0.0002
# incline
python tools/train_ppo.py --cfg ./configs/incline.yaml OUT_DIR ./output/incline_decoder_64_constant_lr_0.0002_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.DECODER_DIMS [64] PPO.LR_POLICY constant PPO.BASE_LR 0.0002
# csr
python tools/train_ppo.py --cfg ./configs/csr.yaml OUT_DIR ./output/csr_constant_lr_0.0002_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. PPO.LR_POLICY constant PPO.BASE_LR 0.0002

# fine tune on test robots
# fine tune by only updating separate PE
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_fine_tune_separate_PE_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/test RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.USE_SEPARATE_PE True PPO.CHECKPOINT_PATH output/ft_separate_PE_zero_init_KL_5_wo_PE+dropout/1409/Unimal-v0.pt
# fine tune by only updating vanilla PE
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_fine_tune_PE_KL_5_wo_dropout/1409 ENV.WALKER_DIR ./unimals_100/test RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 5. PPO.CHECKPOINT_PATH output/ft_PE_zero_init_KL_5_wo_dropout/1409/Unimal-v0.pt
# fine tune the whole model without PE
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_fine_tune_full_model_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/test RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. PPO.CHECKPOINT_PATH output/ft_baseline_KL_5_wo_PE+dropout/1409/Unimal-v0.pt MODEL.FINETUNE.FULL_MODEL True

# train with separate PE after some iterations
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_separate_PE_zero_init_600_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.USE_SEPARATE_PE True MODEL.TRANSFORMER.SEPARATE_PE_UPDATE_ITER 600

# train wo Vecnorm, norm the context dimensions with pre-given range
# ft
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR output/ft_no_vecnorm_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.BASE_CONTEXT_NORM fixed
# csr
python tools/train_ppo.py --cfg ./configs/csr.yaml OUT_DIR output/csr_no_vecnorm_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.OBS_TO_NORM [] MODEL.BASE_CONTEXT_NORM fixed

# normalize over all limbs
# ft
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR output/ft_limb_norm_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.NORM_OVER_LIMB True
# csr
python tools/train_ppo.py --cfg ./configs/csr.yaml OUT_DIR output/csr_limb_norm_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.NORM_OVER_LIMB True

# normalize over all limbs (exclude zero-padding limbs for statistics computation)
# ft
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR output/ft_limb_norm_wo_padding_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.NORM_OVER_LIMB True MODEL.INCLUDE_PADDING_LIMB_IN_NORM False
# csr
python tools/train_ppo.py --cfg ./configs/csr.yaml OUT_DIR output/csr_limb_norm_wo_padding_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.NORM_OVER_LIMB True MODEL.INCLUDE_PADDING_LIMB_IN_NORM False

# train wo Vecnorm, norm both state and context dimensions with pre-given range
# ft
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR output/ft_fixed_range_state+context_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.BASE_CONTEXT_NORM fixed MODEL.OBS_FIX_NORM True
# csr
python tools/train_ppo.py --cfg ./configs/csr.yaml OUT_DIR output/csr_fixed_range_state+context_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 3. MODEL.OBS_TO_NORM [] MODEL.BASE_CONTEXT_NORM fixed MODEL.OBS_FIX_NORM True

# modular PE with fixed norm
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_humanoid_train_TF_fixed_norm/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/humanoid_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.OBS_FIX_NORM True
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_walker_train_TF_fixed_norm/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/walker_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.OBS_FIX_NORM True

# modular PE with different heuristics
# vanilla PE
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_humanoid_train_TF_wo_vecnorm_vanilla_PE/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/humanoid_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM []
# semantic PE
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_humanoid_train_TF_wo_vecnorm_semantic_PE/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/humanoid_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.OBS_TO_NORM [] MODEL.TRANSFORMER.USE_SEMANTIC_PE True
# version with limb norm
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_humanoid_train_TF_limb_norm_vanilla_PE/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/humanoid_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 5. MODEL.NORM_OVER_LIMB True
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/modular_humanoid_train_TF_limb_norm_semantic_PE/1409 ENV_NAME Modular-v0 ENV.WALKER_DIR ./modular/humanoid_train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.USE_SEMANTIC_PE True MODEL.NORM_OVER_LIMB True


# ST
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ST_100M_floor-1409-11-14-01-15-44-14/1409 ENV.WALKER_DIR ./unimals_single_task/floor-1409-11-14-01-15-44-14 RNG_SEED 1409 SAVE_HIST_RATIO True MODEL.TRANSFORMER.EMBEDDING_DROPOUT False

python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_per_node_embed/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.PER_NODE_EMBED 1409

# for MLP
# baseline
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3
# HN
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_HN_new_init_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN
# HN with single value network output
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_HN_new_init_single_value_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN MODEL.MLP.SINGLE_VALUE True
# HN for input
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_HN_only_input_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN MODEL.MLP.HN_INPUT True
# HN for output
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_HN_only_output_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN MODEL.MLP.HN_OUTPUT True
# MLP with shared input layer
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_shared_input_relu_before_mean_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN MODEL.MLP.SHARE_INPUT True
# MLP with input shared init
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_MLP_input_shared_init_256*3_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TYPE mlp PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 256 MODEL.MLP.LAYER_NUM 3 MODEL.MLP.MODE HN MODEL.MLP.SHARE_INPUT_INIT True

# baseline
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_400M_separatePE_uniform_sample_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING uniform_random_strategy MODEL.TRANSFORMER.USE_SEPARATE_PE True PPO.MAX_STATE_ACTION_PAIRS 400000000.
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_train_mutate_UED_recheck_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_mutate_400 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING UED TASK_SAMPLING.ST_PATH output/ST_MLP_train_mutate_constant_lr

# positive value loss
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_debug_1000M_mutate_1000_PVL_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_mutate_1000 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING UED UED.SAMPLER ACCEL PPO.MAX_STATE_ACTION_PAIRS 1000000000.

# expanded with uniform sampling
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_400M_expanded_immediate_children_threshold_2000+uniform_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_expand_1409 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING UED UED.SAMPLER uniform UED.GENERATE_NEW_AGENTS True UED.MUTATION_AGENT_NUM 10 UED.PARENT_SELECT_STRATEGY immediate_children UED.MUTATE_THRESHOLD 2000. PPO.MAX_STATE_ACTION_PAIRS 400000000.

python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_baseline_env_64_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. PPO.NUM_ENVS 64
# uniform sampling
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_400M_mutate_1000_HN+FA_env_256_uniform_sample_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_mutate_1000 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING uniform_random_strategy PPO.MAX_STATE_ACTION_PAIRS 400000000. PPO.NUM_ENVS 256 MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.CONTEXT_ENCODER linear
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_400M_mutate_1000_uniform_sample_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_mutate_1000 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING uniform_random_strategy PPO.MAX_STATE_ACTION_PAIRS 400000000.
# UED
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_debug_baseline_UED_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING UED TASK_SAMPLING.ST_PATH output/ST_MLP_train_mutate_constant_lr
# UED: learning progress
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_400M_mutate_1000_curation_LP_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train_mutate_1000 RNG_SEED 1409 MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None PPO.KL_TARGET_COEF 5. ENV.TASK_SAMPLING UED UED.SAMPLER learning_progress PPO.MAX_STATE_ACTION_PAIRS 400000000. PPO.NUM_ENVS 256

# FA
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_FA+SWAT_RE_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT True MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.CONTEXT_ENCODER linear MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT True MODEL.TRANSFORMER.USE_SWAT_RE True
# HN
python tools/train_ppo.py --cfg ./configs/exploration.yaml OUT_DIR ./output/exploration_new_wo_visitation_HN_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT True MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT True ENV.KEYS_TO_KEEP [] MODEL.TRANSFORMER.EXT_MIX None
# HN+FA
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/ft_train+joint_angle_300_HN+FA_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/joint_angle_300 RNG_SEED 1409 PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.CONTEXT_ENCODER linear
# ablation on HN
python tools/train_ppo.py --cfg ./configs/exploration.yaml OUT_DIR ./ablation/exploration_HN_input+FA_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT True MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.CONTEXT_ENCODER linear MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT True MODEL.TRANSFORMER.HN_DECODER False
python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./ablation/ft_HN_output+FA_KL_5_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 PPO.KL_TARGET_COEF 5. MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT True MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.CONTEXT_ENCODER linear MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT True MODEL.TRANSFORMER.HN_EMBED False

python tools/train_ppo.py --cfg ./configs/exploration.yaml --no_context_in_state OUT_DIR ./ablation/exploration_HN+FA_wo_context_input_KL_3_wo_PE+dropout/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.POS_EMBEDDING None MODEL.TRANSFORMER.EMBEDDING_DROPOUT False PPO.KL_TARGET_COEF 3. MODEL.TRANSFORMER.NODE_DEPTH_IN_CONTEXT True MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.CONTEXT_ENCODER linear MODEL.TRANSFORMER.CHILD_NUM_IN_CONTEXT True PPO.MAX_STATE_ACTION_PAIRS 2e8

python tools/train_ppo.py --cfg ./configs/ft.yaml OUT_DIR ./output/mlp_ft_128*2_KL_5/1409 ENV.WALKER_DIR ./unimals_100/train RNG_SEED 1409 PPO.KL_TARGET_COEF 5. MODEL.MLP.HIDDEN_DIM 128 MODEL.MLP.LAYER_NUM 2


python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_600 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_1200 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_1800 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_2400 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_3000 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_3600 --seed 1409
python tools/evaluate.py --policy_path output/ft_400M_train_KL_5_wo_PE+dropout --test_folder unimals_100/test --policy_name checkpoint_4200 --seed 1409


