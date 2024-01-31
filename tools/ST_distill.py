import os

from .evaluate import 


agents = [x[:-4] for x in os.listdir('data/train/xml')][:10]

for agent in agents:
    os.makedirs(f'data/{agent}', exist_ok=True)
    os.makedirs(f'data/{agent}/xml', exist_ok=True)
    os.makedirs(f'data/{agent}/metadata', exist_ok=True)
    os.system(f'cp data/train/xml/{agent}.xml data/{agent}/xml/')
    os.system(f'cp data/train/metadata/{agent}.json data/{agent}/metadata/')

    for seed in [1409, 1410, 1411]:
        os.system(f'python tools/distill.py --cfg configs/obstacle_compressed.yaml --context_version 1 RNG_SEED {seed} \
            DISTILL.SOURCE obstacle_200M_HN+FA_KL_5_wo_PE+dropout_80k*100/{seed} \
            MODEL.TYPE transformer MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None \
            MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.CONTEXT_ENCODER linear \
            OUT_DIR KD/obstacle_ST_TF_{agent}/{seed} \
            DISTILL.PER_AGENT_SAMPLE_NUM 80000 DISTILL.BASE_LR 0.0003 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 50 \
            DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True \
            ENV.WALKER_DIR data/{agent} DISTILL.SAMPLE_WEIGHT True')

        model_path = f'KD/obstacle_ST_TF_{agent}/{seed}/Unimal-v0.pt'
        agent_path = f'data/{agent}'
        policy_folder = f'KD/obstacle_ST_TF_{agent}/{seed}'
        score = evaluate_model(model_path, agent_path, policy_folder, suffix='ST', deterministic=True, compute_gae=False)
        print (score)

    break
