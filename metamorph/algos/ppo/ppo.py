import os
import time
import pickle
import json
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
from .ued import TaskSampler, mutate_robot

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
        os.system(f'mkdir {cfg.OUT_DIR}/limb_ratio')
        os.system(f'mkdir {cfg.OUT_DIR}/iter_sampled_agents')
        os.system(f'mkdir {cfg.OUT_DIR}/iter_prob')
        os.system(f'mkdir {cfg.OUT_DIR}/ACCEL_score')

        self.ST_performance = None

        if cfg.ENV.TASK_SAMPLING == 'UED':
            self.task_sampler = TaskSampler(
                cfg.ENV.WALKERS, 
                mutation_agent_num=cfg.UED.MUTATION_AGENT_NUM, 
                staleness_score_weight=cfg.UED.STALENESS_WEIGHT, 
                potential_score_EMA_coef=cfg.UED.SCORE_EMA_COEF, 
            )

    def train(self):

        if cfg.UED.USE_VALIDATION:
            # generate mutations of default_100 to initialize the validation set
            valid_agents = []
            for agent in cfg.ENV.WALKERS:
                valid_agents.extend(mutate_robot(agent, cfg.ENV.WALKER_DIR))
            # hack for debug
            # valid_agents = [x[:-4] for x in os.listdir('unimals_100/train_valid_1409_backup/xml') if x not in os.listdir('unimals_100/train/xml')]
            # os.system('rm -r unimals_100/train_valid_1409')
            # os.system('cp -r unimals_100/train_valid_1409_backup unimals_100/train_valid_1409')
            self.valid_meter = TrainMeter(agents=valid_agents)
            self.valid_envs = make_vec_envs(env_type='valid')
            with open(f'{cfg.OUT_DIR}/walkers_valid.json', 'w') as f:
                json.dump(self.valid_meter.agents, f)
            self.save_valid_sampled_agent_seq()

        # save initial train and valid agents
        with open(f'{cfg.OUT_DIR}/walkers_train.json', 'w') as f:
            json.dump(self.train_meter.agents, f)

        self.save_sampled_agent_seq(-1)

        obs = self.envs.reset()
        self.buffer.to(self.device)
        self.start = time.time()

        print ('obs')
        print (type(obs), len(obs))
        for key in obs:
            print (key, obs[key].size())

        self.grad_record = defaultdict(list)
        self.std_record = []

        # do not update separate PE at the beginning
        # if cfg.MODEL.TRANSFORMER.USE_SEPARATE_PE:
        #     self.agent.ac.mu_net.separate_PE_encoder.pe.requires_grad = False
        #     self.agent.ac.v_net.separate_PE_encoder.pe.requires_grad = False

        if cfg.PPO.MAX_ITERS > 1000:
            self.stat_save_freq = 100
        else:
            self.stat_save_freq = 10
        
        if cfg.PPO.MAX_ITERS > 500:
            model_save_freq = 100
        else:
            model_save_freq = 50

        obs_min_record, obs_max_record = [], []
        # a buffer to store each episode's GAE for UED
        episode_value = np.zeros([cfg.PPO.NUM_ENVS, 1000])
        episode_reward = np.zeros([cfg.PPO.NUM_ENVS, 1000])
        episode_timestep = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)
        for cur_iter in range(cfg.PPO.MAX_ITERS):

            # store the agents sampled in each iteration for UED
            self.sampled_agents = []

            # if cfg.MODEL.TRANSFORMER.USE_SEPARATE_PE and cur_iter >= cfg.MODEL.TRANSFORMER.SEPARATE_PE_UPDATE_ITER:
            #     print ('start to tune separate PE')
            #     self.agent.ac.mu_net.separate_PE_encoder.pe.requires_grad = True
            #     self.agent.ac.v_net.separate_PE_encoder.pe.requires_grad = True

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break
            
            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                unimal_ids = self.envs.get_unimal_idx()
                val, act, logp, dropout_mask_v, dropout_mask_mu = self.agent.act(obs, unimal_ids=unimal_ids)
                # limb_logp = self.agent.limb_logp.detach()
                limb_logp = None

                if cfg.PPO.TANH == 'action':
                    next_obs, reward, done, infos = self.envs.step(torch.tanh(act))
                else:
                    next_obs, reward, done, infos = self.envs.step(act)

                # store each episode's value and reward to compute the trajectory GAE
                # episode_value[np.arange(cfg.PPO.NUM_ENVS), episode_timestep] = val.cpu().numpy().ravel()
                # episode_reward[np.arange(cfg.PPO.NUM_ENVS), episode_timestep] = reward.ravel()
                # episode_timestep += 1
                # # check episode end
                # finished_episode_index = np.where(done)[0]
                # for idx in finished_episode_index:
                #     episode_len = episode_timestep[idx]
                #     delta = episode_reward[idx, :(episode_len - 1)] + cfg.PPO.GAMMA * episode_value[idx, 1:episode_len] - episode_value[idx, :(episode_len - 1)]
                #     # the last step's delta need additional process
                #     if 'timeout' in infos[idx]:
                #         delta = np.append(delta, 0.)
                #     else:
                #         delta = np.append(delta, episode_reward[idx, -1] - episode_value[idx, -1])
                #     # compute GAE
                #     gae = delta.copy()
                #     for i in reversed(range(gae.shape[0] - 1)):
                #         gae[i] = delta[i] + cfg.PPO.GAMMA * cfg.PPO.GAE_LAMBDA * gae[i + 1]
                #     learning_potential_score = np.maximum(gae, 0.).mean()
                #     infos[idx]['positive_gae'] = learning_potential_score
                #     # reset the episode record
                #     episode_value[idx] = 0.
                #     episode_reward[idx] = 0.
                #     episode_timestep[idx] = 0

                finished_episode_index = np.where(done)[0]
                self.train_meter.add_ep_info(infos, cur_iter, finished_episode_index)
                # record agents that are done
                finished_agents = [infos[i]['name'] for i in range(len(done)) if done[i]]
                self.sampled_agents.extend(finished_agents)

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

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts, dropout_mask_v, dropout_mask_mu, unimal_ids, limb_logp)
                obs = next_obs

            self.train_meter.update_iter_returns(cur_iter)

            unimal_ids = self.envs.get_unimal_idx()
            next_val = self.agent.get_value(obs, unimal_ids=unimal_ids)
            self.buffer.compute_returns(next_val)
            # self.train_on_batch(cur_iter)

            with open(f'{cfg.OUT_DIR}/iter_sampled_agents/{cur_iter}.pkl', 'wb') as f:
                pickle.dump(self.sampled_agents, f)

            if cfg.ENV.TASK_SAMPLING == 'UED' and cfg.UED.CURATION == 'positive_value_loss':
                # update UED scores of the agents sampled in the current iteration
                self.task_sampler.update_scores(self.buffer)
                with open(f'{cfg.OUT_DIR}/ACCEL_score/{cur_iter}.pkl', 'wb') as f:
                    pickle.dump([self.task_sampler.potential_score, self.task_sampler.staleness_score], f)
                # for agent in set(self.sampled_agents):
                #     agent_id = cfg.ENV.WALKERS.index(agent)
                #     potential_score = np.mean(self.train_meter.agent_meters[agent].ep_positive_gae)
                #     self.agent_manager.potential_score[agent_id] = potential_score
                #     staleness_score = cur_iter
                #     self.agent_manager.staleness_score[agent_id] = staleness_score
            
            if cfg.ENV.TASK_SAMPLING == 'UED' and cfg.UED.GENERATE_NEW_AGENTS:
                # generate new agents by mutating existing ones
                if cur_iter >= 0 and cur_iter % cfg.UED.ADD_NEW_AGENTS_FREQ == 0:
                    agent_scores = []
                    for agent in cfg.ENV.WALKERS:
                        scores = self.train_meter.agent_meters[agent].mean_ep_rews['reward']
                        if len(scores) < 5:
                            agent_scores.append(0.)
                        else:
                            agent_scores.append(np.array(scores[-5:]).mean())
                    new_agents, new_agents_parent = self.agent_manager.generate_new_agents(set(self.sampled_agents), np.array(agent_scores))
                    if len(new_agents) > 0:
                        new_potential_score = [np.mean(self.train_meter.agent_meters[agent].ep_positive_gae) for agent in new_agents_parent]
                        self.agent_manager.initialize_new_agents_score(new_agents, new_potential_score)
                        cfg.ENV.WALKERS.extend(new_agents)
                        num_agents = len(cfg.ENV.WALKERS)
                        print (f'The agent pool now contains {num_agents} agents')
                        with open(f'{cfg.OUT_DIR}/walkers.json', 'w') as f:
                            json.dump(cfg.ENV.WALKERS, f)
                        self.train_meter.add_new_agents(new_agents)
                    else:
                        print ('no new agents added in this iteration')

            # save std record
            with open(os.path.join(cfg.OUT_DIR, 'std.pkl'), 'wb') as f:
                pickle.dump(self.std_record, f)

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
            
            if cur_iter % model_save_freq == 0:
                self.save_model(cur_iter)
            
            if cfg.UED.USE_VALIDATION:

                if cur_iter % cfg.UED.CHECK_VALID_FREQ == 0:
                    print ('evaluate on the validation set')
                    self.valid_envs.reset()
                    for step in range(cfg.UED.VALID_TIMESTEPS):
                        # Sample actions
                        _, act, _, _, _ = self.agent.act(obs)
                        next_obs, reward, done, infos = self.valid_envs.step(act)
                        self.valid_meter.add_ep_info(infos, cur_iter, np.where(done)[0])
                        obs = next_obs
                    self.valid_meter.update_iter_returns(cur_iter)
                    # move converged valid agents to the training set
                    agents_to_move = []
                    for agent in self.valid_meter.agents:
                        agent_meter = self.valid_meter.agent_meters[agent]
                        if len(agent_meter.iter_mean_return) == 0:
                            continue
                        if agent_meter.iter_mean_return[-1] < agent_meter.best_iter_return and cur_iter - agent_meter.best_iter >= 3 * cfg.UED.CHECK_VALID_FREQ:
                            agents_to_move.append(agent)
                    # move the converged validation agents to the training set
                    print ('move the following agents to the training set')
                    print (agents_to_move)
                    self.valid_meter.delete_agents(agents_to_move)
                    self.train_meter.add_new_agents(agents_to_move)
                    cfg.ENV.WALKERS.extend(agents_to_move)
                    # mutate the moved agents to get new validation agents
                    new_valid_agents = []
                    for agent in agents_to_move:
                        new_valid_agents.extend(mutate_robot(agent, cfg.ENV.WALKER_DIR))
                    self.valid_meter.add_new_agents(new_valid_agents)
                    # save the updated validation set
                    with open(f'{cfg.OUT_DIR}/walkers_valid.json', 'w') as f:
                        json.dump(self.valid_meter.agents, f)
                    self.save_valid_sampled_agent_seq()

                if cur_iter % cfg.UED.CHECK_TRAIN_FREQ == 0:
                    uncontrollable_agents = []
                    for agent in self.train_meter.agents:
                        agent_meter = self.train_meter.agent_meters[agent]
                        # check performance convergencce
                        if len(agent_meter.iter_mean_return) == 0:
                            continue
                        if agent_meter.iter_mean_return[-1] < agent_meter.best_iter_return and cur_iter - agent_meter.best_iter >= 3 * cfg.UED.CHECK_TRAIN_FREQ:
                            # TODO: this is not reasonable at the early stage of training
                            if agent_meter.best_iter_return < 500:
                                uncontrollable_agents.append(agent)
                    print ('remove the following agents from the training set')
                    print (uncontrollable_agents)
                    self.train_meter.delete_agents(uncontrollable_agents)
                    for agent in uncontrollable_agents:
                        cfg.ENV.WALKERS.remove(agent)

            with open(f'{cfg.OUT_DIR}/walkers_train.json', 'w') as f:
                json.dump(cfg.ENV.WALKERS, f)
            # sample agents for next iteration
            self.save_sampled_agent_seq(cur_iter)

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        ratio_hist = [[] for _ in range(cfg.PPO.EPOCHS)]
        grad_norm_dict, grad_correlation_dict = defaultdict(list), defaultdict(list)
        limb_ratio_record = []

        std_record = []
        additional_clip_frac_record = []
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

                # limb ratio
                if cfg.SAVE_LIMB_RATIO:
                    limb_ratio = torch.exp(self.actor_critic.limb_logp.detach() - batch["limb_logp_old"]).cpu().numpy()
                    limb_ratio_record.append(limb_ratio)

                if cfg.PPO.KL_TARGET_COEF is not None and approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    self.train_meter.add_train_stat("approx_kl", approx_kl)
                    if cfg.SAVE_HIST_RATIO:
                        with open(os.path.join(cfg.OUT_DIR, 'ratio_hist', f'ratio_hist_{cur_iter}.pkl'), 'wb') as f:
                            pickle.dump(ratio_hist, f)
                    print (f'early stop iter {cur_iter} at epoch {i + 1}/{cfg.PPO.EPOCHS}, batch {j + 1} with approx_kl {approx_kl}')
                    return

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
                std_record.append(np.exp(log_std))

                self.train_meter.add_train_stat("approx_kl", approx_kl)
                self.train_meter.add_train_stat("pi_loss", pi_loss.item())
                self.train_meter.add_train_stat("val_loss", val_loss.item())
                self.train_meter.add_train_stat("ratio", ratio.mean().item())
                self.train_meter.add_train_stat("surr1", surr1.mean().item())
                self.train_meter.add_train_stat("surr2", surr2.mean().item())
                self.train_meter.add_train_stat("clip_frac", clip_frac)

                self.optimizer.step()

        self.std_record.append(std_record)

        if cfg.SAVE_LIMB_RATIO and cur_iter % self.stat_save_freq == 0:
            # save the grad results
            with open(f'{cfg.OUT_DIR}/limb_ratio/{cur_iter}.pkl', 'wb') as f:
                pickle.dump(limb_ratio_record, f)

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
        returns = []

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
                returns.append(infos[0]['episode']['r'])
                episode_count += 1
                if episode_count == 5:
                    break

        env.close()
        # remove annoying meta file created by monitor
        avg_return = int(np.array(returns).mean())
        os.remove(os.path.join(save_dir, f"{xml}_video.meta.json"))
        os.rename(os.path.join(save_dir, f"{xml}_video.mp4"), os.path.join(save_dir, f"{xml}_video_{avg_return}.mp4"))
        return returns
        # return action_history, obs_record, reset_step, returns

    def save_sampled_agent_seq(self, cur_iter):
        agents = self.train_meter.agents
        num_agents = len(agents)

        if num_agents <= 1:
            return

        if cfg.ENV.TASK_SAMPLING == "uniform_random_strategy":
            ep_lens = [1000] * num_agents
            probs = [1000.0 / l for l in ep_lens]

        elif cfg.ENV.TASK_SAMPLING == "balanced_replay_buffer":
            # For a first couple of iterations do uniform sampling to ensure
            # we have good estimate of ep_lens
            if num_agents > 150:
                # follow a different strategy if we use augmented training set
                ep_lens = [len(self.train_meter.agent_meters[agent].ep_len) for agent in agents]
                if min(ep_lens) == 10:
                    print ('start to sample based on ema')
                    if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
                        ep_lens = [
                            np.mean(self.train_meter.agent_meters[agent].ep_len_ema)
                            for agent in agents
                        ]
                    elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
                        ep_lens = [
                            np.mean(self.train_meter.agent_meters[agent].ep_len)
                            for agent in agents
                        ]
                else:
                    ep_lens = [l + 1 for l in ep_lens]
            else:
                if cur_iter < 30:
                    ep_lens = [1000] * num_agents
                else:
                    if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
                        ep_lens = [
                            np.mean(self.train_meter.agent_meters[agent].ep_len_ema)
                            for agent in agents
                        ]
                    elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
                        ep_lens = [
                            np.mean(self.train_meter.agent_meters[agent].ep_len)
                            for agent in agents
                        ]
            probs = [1000.0 / l for l in ep_lens]

        elif cfg.ENV.TASK_SAMPLING == "UED":
            
            if cfg.UED.CURATION == 'positive_value_loss':
                # select new agents for the next learning iteration
                if cur_iter >= int(cfg.PPO.MAX_ITERS / 20.):
                    probs = self.task_sampler.get_sample_probs(cur_iter)
                else:
                    probs = [1. for _ in agents]
            
            elif cfg.UED.SAMPLER == 'regret':

                if self.ST_performance is None:
                    self.ST_performance = {}
                    for agent in agents:
                        with open(f'{cfg.TASK_SAMPLING.ST_PATH}/{agent}/1409/Unimal-v0_results.json', 'r') as f:
                            log = json.load(f)
                        self.ST_performance[agent] = log[agent]['reward']['reward'][-1]

                ep_lens = [len(self.train_meter.agent_meters[agent].ep_len) for agent in agents]
                if min(ep_lens) == 10:
                    print ('start to sample based on UED regret')
                    # TODO: how to deal with negative regret
                    # Issue: training score is computed based on an out-of-date model, 
                    # which may not correctly reflect the current model's performance on a robot
                    probs = [
                        self.ST_performance[agent] - self.train_meter.agent_meters[agent].ep_len_ema 
                        for agent in agents
                    ]
                    probs = [max(p, 0.) for p in probs]
                else:
                    ep_lens = [l + 1 for l in ep_lens]
                    probs = [1000.0 / l for l in ep_lens]
            
            elif cfg.UED.SAMPLER == 'learning_progress':
                ep_lens = [len(self.train_meter.agent_meters[agent].ep_len) for agent in agents]
                if min(ep_lens) == 10:
                    print ('start to sample based on learning progress')
                    LP_score = [self.train_meter.agent_meters[agent].get_learning_speed() for agent in agents]
                    probs = LP_score
                    print (probs)
                else:
                    ep_lens = [l + 1 for l in ep_lens]
                    probs = [1000.0 / l for l in ep_lens]

            else:
                probs = [1. for _ in agents]

        # maybe use softmax here?
        # probs = np.exp(probs)
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = [p / sum(probs) for p in probs]
        with open(f'{cfg.OUT_DIR}/iter_prob/{cur_iter}.pkl', 'wb') as f:
            pickle.dump(probs, f)

        # Estimate approx number of episodes each subproc env can rollout
        avg_ep_len = np.mean([
            np.mean(self.train_meter.agent_meters[agent].ep_len)
            for agent in agents
        ])
        # In the start the arrays will be empty
        if np.isnan(avg_ep_len):
            avg_ep_len = 100
        ep_per_env = cfg.PPO.TIMESTEPS / avg_ep_len
        # Task list size (multiply by 8 as padding)
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, f"sampling_train.json")
        fu.save_json(task_list, path)

    def save_valid_sampled_agent_seq(self):

        num_agents = len(self.valid_meter.agents)
        probs = [1. / num_agents for _ in range(num_agents)]

        ep_per_env = 100
        # Task list size (multiply by 8 as padding)
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, f"sampling_valid.json")
        fu.save_json(task_list, path)