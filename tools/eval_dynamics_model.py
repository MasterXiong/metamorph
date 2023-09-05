import numpy as np
import torch
import os
import pickle

from metamorph.config import cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.envs import *
from metamorph.algos.ppo.dynamics import *

from tools.train_ppo import set_cfg_options

PROPRIOCEPTIVE_OBS_DIM = np.concatenate([list(range(13)), [30, 31], [41, 42]])


def collect_rollout(env, policy, dynamics_model):

    true_obs, pred_obs, dones = [], [], []

    obs = env.reset()
    obs_mask = obs["obs_padding_mask"].bool()
    act_mask = obs["act_padding_mask"].bool()
    num_env = obs['proprioceptive'].shape[0]

    for t in range(1000):
        _, act, _, _, _ = policy.act(obs, return_attention=False, compute_val=False)
        next_obs, _, done, _ = env.step(act)

        act[act_mask] = 0.0
        act = torch.clip(act, -1., 1.)
        with torch.no_grad():
            next_obs_pred = dynamics_model(torch.cat([obs['proprioceptive'].reshape(num_env, 12, -1), act.reshape(num_env, 12, -1)], -1), obs_mask)
        true_obs.append(next_obs['proprioceptive'].reshape(num_env, 12, -1)[:, :, PROPRIOCEPTIVE_OBS_DIM])
        pred_obs.append(next_obs_pred)
        dones.append(torch.Tensor(done))

        obs = next_obs
    
    true_obs = torch.cat(true_obs)
    pred_obs = torch.cat(pred_obs)
    dones = torch.cat(dones).bool().cuda()
    masks = torch.cat([obs_mask for _ in range(1000)])

    scale_factor = 12. / (12. - masks[0].sum().item())
    feat_dim = true_obs.shape[-1]

    square_error = ((pred_obs - true_obs) ** 2 * ~masks[:, :, None])[~dones]
    # rescale the loss by actual limb number
    mse = square_error.mean().item() * scale_factor
    feature_wise_mse = square_error.reshape(-1, feat_dim).mean(dim=0) * scale_factor

    # relative_error = (((pred_obs - true_obs) / true_obs).abs() * ~masks[:, :, None])[~dones]
    # relative_error[torch.isnan(relative_error) | torch.isinf(relative_error)] = 0.
    # mean_relative_error = relative_error.mean().item() * scale_factor
    # feature_wise_relative_error = relative_error.reshape(-1, feat_dim).mean(dim=0) * scale_factor
    feature_wise_std = true_obs.reshape(-1, feat_dim).std(dim=0)
    feature_wise_relative_error = feature_wise_mse.sqrt() / feature_wise_std

    return mse, feature_wise_mse.cpu().numpy(), feature_wise_relative_error.cpu().numpy()



def main(model_path, agent_path, dynamics_model_name='checkpoint_final'):

    agents = [x.split('.')[0] for x in os.listdir(f'{agent_path}/xml')]

    cfg.merge_from_file(f'output/{model_path}/config.yaml')
    cfg.PPO.CHECKPOINT_PATH = f'output/{model_path}/Unimal-v0.pt'
    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_path
    cfg.OUT_DIR = './eval'
    cfg.PPO.NUM_ENVS = 16
    set_cfg_options()

    ppo_trainer = PPO()
    policy = ppo_trainer.agent
    policy.ac.eval()

    # load the dynamics model
    checkpoint = torch.load(f'dynamics_model/{model_path}/{dynamics_model_name}.pt')
    dynamics_model = DynamicsModel(52+2, 17)
    dynamics_model.load_state_dict(checkpoint[0])
    dynamics_model = dynamics_model.cuda()

    test_set = agent_path.split('/')[1]
    os.makedirs(f'eval_dynamics/{model_path}', exist_ok=True)

    try:
        with open(f'eval_dynamics/{model_path}/{test_set}_{dynamics_model_name}.pkl', 'rb') as f:
            scores = pickle.load(f)
    except:
        scores = {}
    
    for agent in agents:
        if agent in scores:
            continue
        print (agent)
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=True, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))
        mse, feature_wise_mse, feature_wise_relative_error = collect_rollout(envs, policy, dynamics_model)
        envs.close()
        
        print (f'MSE: {mse:.4f}')
        print ('feature-wise MSE')
        print (feature_wise_mse)
        print ('feature-wise relative error')
        print (feature_wise_relative_error)
        
        scores[agent] = [mse, feature_wise_mse, feature_wise_relative_error]
        with open(f'eval_dynamics/{model_path}/{test_set}_{dynamics_model_name}.pkl', 'wb') as f:
            pickle.dump(scores, f)



if __name__ == '__main__':
    
    model_path = 'ft_400M_mutate_1000_env_256_curation_PVL_KL_5_wo_PE+dropout/1409'
    agent_path = 'unimals_100/mutate_300'
    main(model_path, agent_path)
