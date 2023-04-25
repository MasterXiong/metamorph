import os
import time
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from metamorph.config import cfg
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.utils import file as fu
from metamorph.utils import model as mu
from metamorph.utils import optimizer as ou
from metamorph.utils.meter import TrainMeter
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

from .buffer import Buffer
from .envs import get_ob_rms
from .envs import make_vec_envs
from .envs import set_ob_rms
from .inherit_weight import restore_from_checkpoint
from .model import ActorCritic
from .model import Agent

class PPO:
    def __init__(self, print_model=True):
        # Create vectorized envs
        self.envs = make_vec_envs()
        self.file_prefix = cfg.ENV_NAME

        self.device = torch.device(cfg.DEVICE)

        # print ('observation_space')
        # print (self.envs.observation_space)
        # print ('action_space')
        # print (self.envs.action_space)
        # print (self.envs)

        self.actor_critic = globals()[cfg.MODEL.ACTOR_CRITIC](
            self.envs.observation_space, self.envs.action_space
        )
        print ('action space')
        print (self.envs.action_space.low)
        print (self.envs.action_space.high)

        # Used while using train_ppo.py
        if cfg.PPO.CHECKPOINT_PATH:
            ob_rms = restore_from_checkpoint(self.actor_critic)
            set_ob_rms(self.envs, ob_rms)

        if print_model:
            #print(self.actor_critic)
            print("Num params: {}".format(mu.num_params(self.actor_critic)))

        self.actor_critic.to(self.device)
        self.agent = Agent(self.actor_critic)

        # Setup experience buffer
        self.buffer = Buffer(self.envs.observation_space, self.envs.action_space.shape)
        # Optimizer for both actor and critic
        if not cfg.MODEL.MLP.ANNEAL_HN_LR:
            self.optimizer = optim.Adam(
                self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS, weight_decay=cfg.PPO.WEIGHT_DECAY
            )
            self.lr_scale = [1. for _ in self.optimizer.param_groups]
        else:
            # reduce the learning rate for HN in MLP
            parameters, self.lr_scale = [], []
            if cfg.MODEL.TYPE == 'mlp' and cfg.MODEL.MLP.HN_INPUT:
                for name, param in self.actor_critic.named_parameters():
                    if 'context_encoder_for_input' in name or 'hnet_input' in name:
                        parameters += [{'params': [param], 'lr': cfg.PPO.BASE_LR}]
                        self.lr_scale.append(1. / cfg.MODEL.MAX_LIMBS)
                    else:
                        parameters += [{'params': [param]}]
                        self.lr_scale.append(1.)
                    print (name, self.lr_scale[-1])
            self.optimizer = optim.Adam(
                parameters, lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS
            )

        self.train_meter = TrainMeter()
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tensorboard"))
        #obs = self.envs.reset()
        #self.writer.add_graph(self.actor_critic, obs)
        # Get the param name for log_std term, can vary depending on arch
        for name, param in self.actor_critic.state_dict().items():
            if "log_std" in name:
                self.log_std_param = name
                break

        # for name, weight in self.actor_critic.named_parameters():
        #     print(name, weight.requires_grad)

        self.fps = 0
        # os.system(f'mkdir {cfg.OUT_DIR}/ratio_hist')
        os.system(f'mkdir {cfg.OUT_DIR}/grad')

    def train(self):
        self.save_sampled_agent_seq(0)
        obs = self.envs.reset()
        self.buffer.to(self.device)
        self.start = time.time()

        print ('obs')
        print (type(obs), len(obs))
        for key in obs:
            print (key, obs[key].size())
                
        # old_values = self.agent.ac.v_net.context_embed.bias.detach().clone()

        self.grad_record = defaultdict(list)

        for cur_iter in range(cfg.PPO.MAX_ITERS):

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break
            
            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)

            # print (self.agent.ac.v_net.hnet_input_weight.weight.data[:6, :6])

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                if cfg.MODEL.TRANSFORMER.USE_SEPARATE_PE or cfg.MODEL.TRANSFORMER.PER_NODE_EMBED:
                    unimal_ids = self.envs.get_unimal_idx()
                else:
                    unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
                val, act, logp, dropout_mask_v, dropout_mask_mu = self.agent.act(obs, unimal_ids=unimal_ids)

                next_obs, reward, done, infos = self.envs.step(act)

                self.train_meter.add_ep_info(infos)

                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device,
                )
                timeouts = torch.tensor(
                    [[0.0] if "timeout" in info.keys() else [1.0] for info in infos],
                    dtype=torch.float32,
                    device=self.device,
                )

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts, dropout_mask_v, dropout_mask_mu, unimal_ids)
                obs = next_obs

            if cfg.MODEL.TRANSFORMER.USE_SEPARATE_PE or cfg.MODEL.TRANSFORMER.PER_NODE_EMBED:
                unimal_ids = self.envs.get_unimal_idx()
            else:
                unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
            next_val = self.agent.get_value(obs, unimal_ids=unimal_ids)
            self.buffer.compute_returns(next_val)
            self.train_on_batch(cur_iter)
            self.save_sampled_agent_seq(cur_iter)

            self.train_meter.update_mean()
            if len(self.train_meter.mean_ep_rews["reward"]):
                cur_rew = self.train_meter.mean_ep_rews["reward"][-1]
                self.writer.add_scalar(
                    'Reward', cur_rew, self.env_steps_done(cur_iter)
                )
            if (
                cur_iter >= 0
                and cur_iter % cfg.LOG_PERIOD == 0
                and cfg.LOG_PERIOD > 0
            ):
                self._log_stats(cur_iter)

                file_name = "{}_results.json".format(self.file_prefix)
                path = os.path.join(cfg.OUT_DIR, file_name)
                self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
                stats = self.train_meter.get_stats()
                stats["fps"] = self.fps
                fu.save_json(stats, path)
                print (cfg.OUT_DIR)
            
            if cur_iter % 100 == 0:
                self.save_model(cur_iter)

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        ratio_hist = [[] for _ in range(cfg.PPO.EPOCHS)]
        grad_norm_dict, grad_correlation_dict = defaultdict(list), defaultdict(list)

        for i in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for j, batch in enumerate(batch_sampler):
                # Reshape to do in a single forward pass for all steps
                val, _, logp, ent, _, _ = self.actor_critic(batch["obs"], batch["act"], \
                    dropout_mask_v=batch['dropout_mask_v'], \
                    dropout_mask_mu=batch['dropout_mask_mu'], \
                    unimal_ids=batch['unimal_ids'])
                clip_ratio = cfg.PPO.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()

                if cfg.PPO.KL_TARGET_COEF is not None and approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    self.train_meter.add_train_stat("approx_kl", approx_kl)
                    if cfg.SAVE_HIST_RATIO:
                        with open(os.path.join(cfg.OUT_DIR, 'ratio_hist', f'ratio_hist_{cur_iter}.pkl'), 'wb') as f:
                            pickle.dump(ratio_hist, f)
                    print (f'early stop iter {cur_iter} at epoch {i + 1}/{cfg.PPO.EPOCHS}, batch {j + 1} with approx_kl {approx_kl}')
                    return
                    # TODO: change return to finishing the iterations

                if cfg.SAVE_HIST_RATIO:
                    clipped_ratio = torch.clamp(ratio, 0., 2.0).detach()
                    hist, _ = np.histogram(clipped_ratio.cpu().numpy(), 100, range=(0., 2.))
                    ratio_hist[i].append(hist)

                surr1 = ratio * batch["adv"]

                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                clip_frac = (ratio != surr2).float().mean().item()
                surr2 *= batch["adv"]

                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                if cfg.PER_LIMB_GRAD:
                    # compute the grad of each limb's action logp
                    self.optimizer.zero_grad()
                    action_logp = self.actor_critic.limb_logp
                    pi_loss.backward(retain_graph=True, inputs=action_logp)
                    action_logp_grad = action_logp.grad.detach()
                    # print ('backward action logp grad')
                    # print (action_logp_grad[0])
                    # print ('analytic action logp grad')
                    # print (-surr1[0] / 5120.)
                    per_limb_grad = defaultdict(list)
                    for k in range(cfg.MODEL.MAX_LIMBS):
                        self.optimizer.zero_grad()
                        filter_grad = torch.zeros(action_logp.size()).cuda()
                        filter_grad[:, (k*2):(k*2+2)] = action_logp_grad[:, (k*2):(k*2+2)]
                        action_logp.backward(gradient=filter_grad, retain_graph=True)
                        for name, param in self.actor_critic.named_parameters():
                            if name == 'log_std' or 'v_net' in name:
                                continue
                            grad = param.grad.clone()
                            per_limb_grad[name].append(grad)
                    # self.optimizer.zero_grad()
                    # action_logp.backward(gradient=action_logp_grad, retain_graph=True)
                    # per_limb_grad = self.actor_critic.mu_net.decoder[0].weight.grad.clone()
                    for name in per_limb_grad:
                        all_limb_grad = torch.stack(per_limb_grad[name], dim=0).reshape(cfg.MODEL.MAX_LIMBS, -1)
                        # compute the norm of each limb's grad
                        norm = all_limb_grad.norm(dim=1)
                        grad_norm_dict[name].append(norm.detach().cpu().numpy())
                        # compute the correlation between different limbs' gradient
                        correlation = torch.corrcoef(all_limb_grad).detach().cpu().numpy()
                        grad_correlation_dict[name].append(correlation)

                self.optimizer.zero_grad()

                loss = val_loss * cfg.PPO.VALUE_COEF
                loss += pi_loss
                loss += -ent * cfg.PPO.ENTROPY_COEF
                loss.backward()

                # Log training stats
                norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM
                )
                # if i == 0 and j == 0:
                #     print (norm)
                # print (f'epoch {i}, batch {j}, gradient norm: {norm}, approx_kl: {approx_kl}')
                self.train_meter.add_train_stat("grad_norm", norm.item())

                log_std = (
                    self.actor_critic.state_dict()[self.log_std_param].cpu().numpy()[0]
                )
                std = np.mean(np.exp(log_std))
                self.train_meter.add_train_stat("std", float(std))

                self.train_meter.add_train_stat("approx_kl", approx_kl)
                self.train_meter.add_train_stat("pi_loss", pi_loss.item())
                self.train_meter.add_train_stat("val_loss", val_loss.item())
                self.train_meter.add_train_stat("ratio", ratio.mean().item())
                self.train_meter.add_train_stat("surr1", surr1.mean().item())
                self.train_meter.add_train_stat("surr2", surr2.mean().item())
                self.train_meter.add_train_stat("clip_frac", clip_frac)

                self.optimizer.step()

        if cfg.PER_LIMB_GRAD:
            # save the grad results
            with open(f'{cfg.OUT_DIR}/grad/{cur_iter}.pkl', 'wb') as f:
                pickle.dump([grad_norm_dict, grad_correlation_dict], f)

        if cfg.SAVE_HIST_RATIO:
            with open(os.path.join(cfg.OUT_DIR, 'ratio_hist', f'ratio_hist_{cur_iter}.pkl'), 'wb') as f:
                pickle.dump(ratio_hist, f)

        # Save weight histogram
        if cfg.SAVE_HIST_WEIGHTS:
            for name, weight in self.actor_critic.named_parameters():
                self.writer.add_histogram(name, weight, cur_iter)
                try:
                    self.writer.add_histogram(f"{name}.grad", weight.grad, cur_iter)
                except NotImplementedError:
                    # If layer does not have .grad move on
                    continue

    def save_model(self, cur_iter, path=None):
        if not path:
            path = os.path.join(cfg.OUT_DIR, self.file_prefix + ".pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], path)
        checkpoint_path = os.path.join(cfg.OUT_DIR, f"checkpoint_{cur_iter}.pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], checkpoint_path)

    def _log_stats(self, cur_iter):
        self._log_fps(cur_iter)
        self.train_meter.log_stats()

    def _log_fps(self, cur_iter, log=True):
        env_steps = self.env_steps_done(cur_iter)
        end = time.time()
        self.fps = int(env_steps / (end - self.start))
        if log:
            print(
                "Updates {}, num timesteps {}, FPS {}".format(
                    cur_iter, env_steps, self.fps
                )
            )

    def env_steps_done(self, cur_iter):
        return (cur_iter + 1) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS

    def save_rewards(self, path=None, hparams=None):
        if not path:
            file_name = "{}_results.json".format(self.file_prefix)
            path = os.path.join(cfg.OUT_DIR, file_name)

        self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
        stats = self.train_meter.get_stats()
        stats["fps"] = self.fps
        fu.save_json(stats, path)

        # Save hparams when sweeping
        if hparams:
            # Remove hparams which are of type list as tensorboard complains
            # on saving it's not a supported type.
            hparams_to_save = {
                k: v for k, v in hparams.items() if not isinstance(v, list)
            }
            final_env_reward = np.mean(stats["__env__"]["reward"]["reward"][-100:])
            self.writer.add_hparams(hparams_to_save, {"reward": final_env_reward})

        self.writer.close()

    def save_video(self, save_dir, xml=None):
        env = make_vec_envs(training=False, norm_rew=False, save_video=True, xml_file=xml)
        set_ob_rms(env, get_ob_rms(self.envs))
        
        env = VecVideoRecorder(
            env,
            save_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=cfg.PPO.VIDEO_LENGTH,
            # file_prefix=self.file_prefix,
            file_prefix=xml,
        )
        
        obs = env.reset()
        # obs_record = {"body_xpos": [], "body_xvelp": [], "body_xvelr": [],"qpos": [], "qvel": []}
        # action_history = []
        # reset_step = []
        # returns = []

        episode_count = 0
        for t in range(cfg.PPO.VIDEO_LENGTH + 1):

            # obs_reshape = obs['proprioceptive'].reshape(-1, 52)[(1. - obs["obs_padding_mask"]).bool().reshape(-1)].detach().cpu().numpy()
            # obs_record['body_xpos'].append(obs_reshape[:, :3])
            # obs_record['body_xvelp'].append(obs_reshape[:, 3:6])
            # obs_record['body_xvelr'].append(obs_reshape[:, 6:9])
            # obs_record['qpos'].append(obs_reshape[:, [30, 39]])
            # obs_record['qvel'].append(obs_reshape[:, [31, 40]])
            
            _, act, _, _, _ = self.agent.act(obs)
            obs, _, _, infos = env.step(act)

            # x = act[(1. - obs["act_padding_mask"]).bool()].detach().cpu().numpy().ravel()
            # action_history.append(x)

            if 'episode' in infos[0]:
                # reset_step.append(t)
                print (infos[0]['episode']['r'])
                # returns.append(infos[0]['episode']['r'])
                episode_count += 1
                if episode_count == 5:
                    break

        env.close()
        # remove annoying meta file created by monitor
        os.remove(os.path.join(save_dir, f"{xml}_video.meta.json"))
        # return action_history, obs_record, reset_step, returns

    def save_sampled_agent_seq(self, cur_iter):
        num_agents = len(cfg.ENV.WALKERS)

        if num_agents <= 1:
            return

        if cfg.ENV.TASK_SAMPLING == "uniform_random_strategy":
            ep_lens = [1000] * num_agents
        elif cfg.ENV.TASK_SAMPLING == "balanced_replay_buffer":
            # For a first couple of iterations do uniform sampling to ensure
            # we have good estimate of ep_lens
            if cur_iter < 30:
                ep_lens = [1000] * num_agents
            else:
                if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
                    ep_lens = [
                        np.mean(self.train_meter.agent_meters[agent].ep_len_ema)
                        for agent in cfg.ENV.WALKERS
                    ]
                elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
                    ep_lens = [
                        np.mean(self.train_meter.agent_meters[agent].ep_len)
                        for agent in cfg.ENV.WALKERS
                    ]

        probs = [1000.0 / l for l in ep_lens]
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = [p / sum(probs) for p in probs]

        # Estimate approx number of episodes each subproc env can rollout
        avg_ep_len = np.mean([
            np.mean(self.train_meter.agent_meters[agent].ep_len)
            for agent in cfg.ENV.WALKERS
        ])
        # In the start the arrays will be empty
        if np.isnan(avg_ep_len):
            avg_ep_len = 100
        ep_per_env = cfg.PPO.TIMESTEPS / avg_ep_len
        # Task list size (multiply by 8 as padding)
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, "sampling.json")
        fu.save_json(task_list, path)
