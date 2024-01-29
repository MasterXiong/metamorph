import os


train_agents = [x[:-4] for x in os.listdir('data/train/xml')][:10]

for agent in agents:
    os.makedirs(f'data/{agent}', exist_ok=True)
    os.makedirs(f'data/{agent}/xml', exist_ok=True)
    os.makedirs(f'data/{agent}/metadata', exist_ok=True)
    os.system(f'cp data/train/xml/{agent}.pkl data/{agent}/xml/')
    os.system(f'cp data/train/metadata/{agent}.json data/{agent}/metadata/')

    for seed in [1409, 1410, 1411]:
        os.system(f'python tools/distill.py --cfg configs/ft_compressed.yaml --context_version 1 RNG_SEED {seed} \
            DISTILL.SOURCE ft_HN+FA_KL_5_wo_PE+dropout_80k*100/{seed} \
            MODEL.TYPE transformer MODEL.TRANSFORMER.EMBEDDING_DROPOUT False MODEL.TRANSFORMER.POS_EMBEDDING None \
            MODEL.TRANSFORMER.FIX_ATTENTION True MODEL.TRANSFORMER.HYPERNET True MODEL.TRANSFORMER.CONTEXT_ENCODER linear \
            OUT_DIR distilled_policy/ft_MT_modumorph_to_ST_modumorph_compressed_lr_3e-4_expert_size_80k_weighted_loss_{agent}/{seed} \
            DISTILL.PER_AGENT_SAMPLE_NUM 80000 DISTILL.BASE_LR 0.0003 DISTILL.SAVE_FREQ 10 DISTILL.EPOCH_NUM 50 \
            DISTILL.GRAD_NORM 0.5 DISTILL.LOSS_TYPE KL DISTILL.BALANCED_LOSS True \
            ENV.WALKER_DIR data/{agent} DISTILL.SAMPLE_WEIGHT True')
