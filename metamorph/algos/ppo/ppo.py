import os
import time

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

        print ('observation_space')
        print (self.envs.observation_space)
        print ('action_space')
        print (self.envs.action_space)
        print (self.envs)

        self.actor_critic = globals()[cfg.MODEL.ACTOR_CRITIC](
            self.envs.observation_space, self.envs.action_space
        )

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
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS
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

        for cur_iter in range(cfg.PPO.MAX_ITERS):

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break

            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr)

            # print ('state embedding weight')
            # print (self.agent.ac.v_net.limb_embed.bias[:8])
            # print ('context embedding weight')
            # print (self.agent.ac.v_net.context_embed.bias[:8])

            # new_values = self.agent.ac.v_net.context_embed.bias.detach().clone()
            # print (self.agent.ac.v_net.context_embed.bias.requires_grad)
            # if cfg.MODEL.TRANSFORMER.HYPERNET:
            #     print ('***hypernet grad***')
            #     # print (self.agent.ac.v_net.hnet_weight.weight.grad)
            #     print ('parameter value')
            #     print (self.agent.ac.v_net.hnet_weight.weight.data[0, :4])
            #     print ('parameter grad')
            #     try:
            #         print (self.agent.ac.v_net.hnet_weight.weight.grad[0, :4])
            #         print ('  non-zero grad fraction', (self.agent.ac.v_net.hnet_weight.weight.grad != 0.).float().mean())
            #     except:
            #         print ('  none grad')
            #     # print ('is leaf', self.agent.ac.v_net.hnet_weight.weight.is_leaf)
            #     # print (self.agent.ac.v_net.hnet_weight.bias.grad)
            #     # print ('context encoder grad')
            #     # print (self.agent.ac.v_net.context_encoder.linear1.weight.grad[0, :8])
            #     # print (self.agent.ac.v_net.context_encoder.linear1.bias.grad)
            #     print ('***context embed grad***')
            #     print ('parameter value')
            #     print (self.agent.ac.v_net.context_embed.weight.data[0, :4])
            #     print ('parameter grad')
            #     try:
            #         print (self.agent.ac.v_net.context_embed.weight.grad[0, :4])
            #         print ('  non-zero grad fraction', (self.agent.ac.v_net.context_embed.weight.grad != 0.).float().mean())
            #     except:
            #         print ('  none grad')
            # print ('***state embed grad***')
            # print ('parameter value')
            # print (self.agent.ac.v_net.limb_embed.weight.data[0, :4])
            # print ('parameter grad')
            # try:
            #     print (self.agent.ac.v_net.limb_embed.weight.grad[0, :4])
            #     print ('  non-zero grad fraction', (self.agent.ac.v_net.limb_embed.weight.grad != 0.).float().mean())
            # except:
            #     print ('  none grad')
            # print ('\n')
            # print ('is leaf', self.agent.ac.v_net.limb_embed.weight.is_leaf)
            # print (self.agent.ac.v_net.context_embed.bias.grad)
            # print ('number of bias changed in context embedding bias', (new_values != old_values).sum())
            # old_values = new_values
            # print (self.agent.ac.v_net.context_embed.weight.data[0, :8])

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                val, act, logp, dropout_mask_v, dropout_mask_mu = self.agent.act(obs)
                #print (obs['proprioceptive'].size())

                next_obs, reward, done, infos = self.envs.step(act)

                # if (next_obs['context'][0] != obs['context'][0]).sum() != 0:
                #     print (f'context change at step {step} with done={done[0]}')
                #     print (next_obs['context'][0])

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

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts, dropout_mask_v, dropout_mask_mu)
                obs = next_obs

            next_val = self.agent.get_value(obs)
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
                self.save_model(cur_iter)

                file_name = "{}_results.json".format(self.file_prefix)
                path = os.path.join(cfg.OUT_DIR, file_name)
                self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
                stats = self.train_meter.get_stats()
                stats["fps"] = self.fps
                fu.save_json(stats, path)
                print (cfg.OUT_DIR)

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for i in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for j, batch in enumerate(batch_sampler):
                # Reshape to do in a single forward pass for all steps
                val, _, logp, ent, _, _ = self.actor_critic(batch["obs"], batch["act"], dropout_mask_v=batch['dropout_mask_v'], dropout_mask_mu=batch['dropout_mask_mu'])
                clip_ratio = cfg.PPO.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()

                if cfg.PPO.KL_TARGET_COEF is not None and approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    self.train_meter.add_train_stat("approx_kl", approx_kl)
                    print (f'early stop iter {cur_iter} at epoch {i + 1}/{cfg.PPO.EPOCHS}, batch {j + 1} with approx_kl {approx_kl}')
                    return

                surr1 = ratio * batch["adv"]

                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
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

                self.optimizer.zero_grad()

                loss = val_loss * cfg.PPO.VALUE_COEF
                loss += pi_loss
                loss += -ent * cfg.PPO.ENTROPY_COEF
                loss.backward()

                # check gradient norm of each module
                # if i == 0 and j == 0:
                #     print ('grad norm')
                #     norms = []
                #     for name, p in self.actor_critic.named_parameters():
                #         if 'hnet' in name or 'decoder' in name:
                #             g = p.grad
                #             if g is not None:
                #                 grad_norm = torch.norm(g.detach(), 2.)
                #                 norms.append(grad_norm)
                #                 print ('  ', name, grad_norm)
                #             else:
                #                 print ('  ', name, g)
                    # print (f'  total_norm: {torch.norm(torch.stack(norms), 2.)}')

                # Log training stats
                norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM
                )
                print (f'epoch {i}, batch {j}, gradient norm: {norm}, approx_kl: {approx_kl}')
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

                self.optimizer.step()

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

    def save_video(self, save_dir):
        env = make_vec_envs(training=False, norm_rew=False, save_video=True,)
        set_ob_rms(env, get_ob_rms(self.envs))
        
        env = VecVideoRecorder(
            env,
            save_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=cfg.PPO.VIDEO_LENGTH,
            file_prefix=self.file_prefix,
        )
        
        obs = env.reset()

        for _ in range(cfg.PPO.VIDEO_LENGTH + 1):
            _, act, _ = self.agent.act(obs)
            obs, _, _, infos = env.step(act)
            if 'episode' in infos[0]:
                print (infos[0]['episode']['r'])

        env.close()
        # remove annoying meta file created by monitor
        os.remove(os.path.join(save_dir, "{}_video.meta.json".format(self.file_prefix)))

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