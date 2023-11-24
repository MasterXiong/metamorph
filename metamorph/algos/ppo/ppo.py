import os
import time
import pickle
import json
from collections import defaultdict
from multiprocessing import Pool
import copy

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
from .envs import get_ob_rms, get_ret_rms
from .envs import make_vec_envs
from .envs import set_ob_rms, set_ret_rms
from .inherit_weight import restore_from_checkpoint
from .model import ActorCritic
from .model import Agent
from .ued import TaskSampler, mutate_robot, sample_new_robot, mutate_single_robot

PROPRIOCEPTIVE_OBS_DIM = np.concatenate([list(range(13)), [30, 31], [41, 42]])


class PPO:
    def __init__(self, print_model=True):
        # Create vectorized envs
        # if cfg.PPO.CHECKPOINT_PATH:
        #     # do not update normalizer when fine-tuning
        #     self.envs = make_vec_envs(training=False)
        # else:
        self.envs = make_vec_envs()
        self.file_prefix = cfg.ENV_NAME

        # use learned model for rollout step
        if cfg.DYNAMICS.MODEL_STEP:
            checkpoint = torch.load(cfg.DYNAMICS.MODEL_PATH)
            self.dynamics_model = DynamicsModel(52+2, 17)
            self.dynamics_model.load_state_dict(checkpoint[0])
            self.dynamics_model = self.dynamics_model.cuda()

        self.device = torch.device(cfg.DEVICE)

        self.actor_critic = globals()[cfg.MODEL.ACTOR_CRITIC](
            self.envs.observation_space, self.envs.action_space
        )
        print ('action space')
        print (self.envs.action_space.low)
        print (self.envs.action_space.high)

        if cfg.PPO.CHECKPOINT_PATH:
            obs_rms, ret_rms, optimizer_state = restore_from_checkpoint(self.actor_critic)
            set_ob_rms(self.envs, obs_rms)
            if ret_rms:
                set_ret_rms(self.envs, ret_rms)

        if print_model:
            #print(self.actor_critic)
            print("Num params: {}".format(mu.num_params(self.actor_critic)))

        self.actor_critic.to(self.device)
        self.agent = Agent(self.actor_critic)

        # Setup experience buffer
        self.buffer = Buffer(self.envs.observation_space, self.envs.action_space.shape)

        # Optimizer for both actor and critic
        # if not cfg.MODEL.MLP.ANNEAL_HN_LR:
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS, weight_decay=cfg.PPO.WEIGHT_DECAY
        )
        self.lr_scale = [1. for _ in self.optimizer.param_groups]
        # else:
        #     # reduce the learning rate for HN in MLP
        #     parameters, self.lr_scale = [], []
        #     if cfg.MODEL.TYPE == 'mlp' and cfg.MODEL.MLP.HN_INPUT:
        #         for name, param in self.actor_critic.named_parameters():
        #             if 'context_encoder_for_input' in name or 'hnet_input' in name:
        #                 parameters += [{'params': [param], 'lr': cfg.PPO.BASE_LR}]
        #                 self.lr_scale.append(1. / cfg.MODEL.MAX_LIMBS)
        #             else:
        #                 parameters += [{'params': [param]}]
        #                 self.lr_scale.append(1.)
        #             print (name, self.lr_scale[-1])
        #     self.optimizer = optim.Adam(
        #         parameters, lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS
        #     )
        # load optimizer state
        if cfg.PPO.CHECKPOINT_PATH and cfg.PPO.CONTINUE_TRAINING:
            self.optimizer.load_state_dict(optimizer_state)

        self.train_meter = TrainMeter()
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tensorboard"))
        #obs = self.envs.reset()
        #self.writer.add_graph(self.actor_critic, obs)
        # Get the param name for log_std term, can vary depending on arch
        for name, param in self.actor_critic.state_dict().items():
            if "log_std" in name:
                self.log_std_param = name
                break

        self.fps = 0
        os.system(f'mkdir {cfg.OUT_DIR}/iter_prob')
        os.system(f'mkdir {cfg.OUT_DIR}/proxy_score')
        os.system(f'mkdir {cfg.OUT_DIR}/weight_stats')

        self.ST_performance = None
        self.last_probs = None

        if cfg.ENV.TASK_SAMPLING == 'UED':
            self.task_sampler = TaskSampler(
                cfg.ENV.WALKERS, 
                mutation_agent_num=cfg.UED.GENERATION_NUM, 
                staleness_score_weight=cfg.UED.STALENESS_WEIGHT, 
                potential_score_EMA_coef=cfg.UED.SCORE_EMA_COEF, 
            )

    def train(self):

        if cfg.UED.USE_VALIDATION:
            # generate mutations of default_100 to initialize the validation set
            valid_agents = []
            with Pool(processes=50) as pool:
                mutated_valid_agents = pool.starmap(mutate_robot, [(agent, cfg.ENV.WALKER_DIR) for agent in cfg.ENV.WALKERS])
            for agents in mutated_valid_agents:
                valid_agents.extend(agents)
            # hack for debug
            # valid_agents = [x[:-4] for x in os.listdir('unimals_100/random_validation_1409_backup/xml') if x not in os.listdir('unimals_100/random_100_v2/xml')]
            # os.system(f'rm -r {cfg.ENV.WALKER_DIR}')
            # os.system(f'cp -r unimals_100/random_validation_1409_backup {cfg.ENV.WALKER_DIR}')
            self.valid_meter = TrainMeter(agents=valid_agents)
            self.valid_envs = make_vec_envs(training=False, norm_rew=False, env_type='valid')
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
        num_env = obs['proprioceptive'].shape[0]
        limb_obs_dim = obs['proprioceptive'].shape[1] // cfg.MODEL.MAX_LIMBS

        print ('obs')
        print (type(obs), len(obs))
        for key in obs:
            print (key, obs[key].size())

        if cfg.PPO.MAX_ITERS > 1000:
            self.stat_save_freq = 100
        else:
            self.stat_save_freq = 10
        
        if cfg.PPO.MAX_ITERS > 1000:
            model_save_freq = 100
        else:
            model_save_freq = 50

        if cfg.PPO.CHECKPOINT_PATH and cfg.PPO.CONTINUE_TRAINING:
            iter_start = int(cfg.PPO.CHECKPOINT_PATH[:-3].split('_')[-1])
        else:
            iter_start = 0

        print ('Start training ...')
        for cur_iter in range(iter_start, cfg.PPO.MAX_ITERS):

            # save the robots and processes that have mjstep error
            process_with_error = []
            robot_with_error = []

            # if cfg.MODEL.MLP.HN_INPUT:
            #     mu_net_input_mean = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS).cuda()
            #     mu_net_input_std = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS).cuda()
            #     v_net_input_mean = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS).cuda()
            #     v_net_input_std = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS).cuda()
            #     mu_context_embedding = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE).cuda()
            #     v_context_embedding = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS, cfg.MODEL.TRANSFORMER.CONTEXT_EMBED_SIZE).cuda()
            # obs_mean = torch.zeros(limb_obs_dim).cuda()
            # obs_std = torch.zeros(limb_obs_dim).cuda()
            # if cfg.MODEL.MLP.HN_OUTPUT:
            #     mu_net_output_mean = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS)
            #     mu_net_output_std = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS)
            #     v_net_output_mean = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS)
            #     v_net_output_std = torch.zeros(len(cfg.ENV.WALKERS), cfg.MODEL.MAX_LIMBS)

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break
            
            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                unimal_ids = self.envs.get_unimal_idx()
                val, act, logp = self.agent.act(obs, unimal_ids=unimal_ids)
                # obs_mean += obs["proprioceptive"].reshape(-1, limb_obs_dim).mean(dim=0)
                # obs_std += obs["proprioceptive"].reshape(-1, limb_obs_dim).std(dim=0)

                # if cfg.MODEL.MLP.HN_INPUT:
                #     mu_net_input_mean[unimal_ids] = self.agent.ac.mu_net.input_weight.mean(dim=(2, 3)).detach()
                #     mu_net_input_std[unimal_ids] = self.agent.ac.mu_net.input_weight.std(dim=(2, 3)).detach()
                #     v_net_input_mean[unimal_ids] = self.agent.ac.v_net.input_weight.mean(dim=(2, 3)).detach()
                #     v_net_input_std[unimal_ids] = self.agent.ac.v_net.input_weight.std(dim=(2, 3)).detach()
                #     mu_context_embedding[unimal_ids] = self.agent.ac.mu_net.context_embedding_input.detach()
                #     v_context_embedding[unimal_ids] = self.agent.ac.v_net.context_embedding_input.detach()

                # if cfg.MODEL.MLP.HN_OUTPUT:
                #     mu_net_output_mean[unimal_ids] = self.agent.ac.mu_net.output_weight.mean(dim=(2, 3)).detach().cpu()
                #     mu_net_output_std[unimal_ids] = self.agent.ac.mu_net.output_weight.std(dim=(2, 3)).detach().cpu()
                #     v_net_output_mean[unimal_ids] = self.agent.ac.v_net.output_weight.mean(dim=(2, 3)).detach().cpu()
                #     v_net_output_std[unimal_ids] = self.agent.ac.v_net.output_weight.std(dim=(2, 3)).detach().cpu()

                if cfg.DYNAMICS.MODEL_STEP:
                    clipped_act = torch.clip(act, -1., 1.)
                    clipped_act[act_mask] = 0.0
                    next_obs = copy.deepcopy(obs)
                    with torch.no_grad():
                        obs_pred = self.dynamics_model(torch.cat([obs['proprioceptive'].reshape(num_env, 12, -1), clipped_act.reshape(num_env, 12, -1)], -1), obs['obs_padding_mask'])
                    next_obs['proprioceptive'] = next_obs['proprioceptive'].reshape(num_env, 12, -1)
                    next_obs['proprioceptive'][:, :, PROPRIOCEPTIVE_OBS_DIM] = obs_pred
                    next_obs['proprioceptive'] = next_obs['proprioceptive'].reshape(num_env, -1)
                    # TODO: hack the reward, done and infos
                else:
                    if cfg.PPO.TANH == 'action':
                        next_obs, reward, done, infos = self.envs.step(torch.tanh(act))
                    else:
                        next_obs, reward, done, infos = self.envs.step(act)

                for process_id, info in enumerate(infos):
                    if info['mj_step_error']:
                        with open(f'{cfg.OUT_DIR}/mjstep_error.txt', 'a') as f:
                            print (f'iter {cur_iter}, step {step}, process {process_id}', file=f)
                            print (info['mj_step_error'], file=f)
                            print (info, file=f)
                            print ('action: ', file=f)
                            print (act[process_id], file=f)
                            print ('action mask', file=f)
                            print (obs['act_padding_mask'][process_id], file=f)
                            # limb_obs = next_obs['proprioceptive'][process_id, :52].detach().cpu().numpy().ravel()
                            # print (limb_obs, file=f)
                        process_with_error.append(process_id)
                        robot_with_error.append(info['name'])

                finished_episode_index = np.where(done)[0]
                self.train_meter.add_ep_info(infos, cur_iter, finished_episode_index)
                # record agents that are done
                # finished_agents = [infos[i]['name'] for i in range(len(done)) if done[i]]
                # self.sampled_agents.extend(finished_agents)

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

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts, unimal_ids)
                obs = next_obs

            # if not cfg.MODEL.MLP.HN_INPUT:
            #     mu_net_input_mean = self.agent.ac.mu_net.input_layer.weight.data.permute(1, 0).reshape(12, -1).mean(dim=1)
            #     mu_net_input_std = self.agent.ac.mu_net.input_layer.weight.data.permute(1, 0).reshape(12, -1).std(dim=1)
            #     v_net_input_mean = self.agent.ac.v_net.input_layer.weight.data.permute(1, 0).reshape(12, -1).mean(dim=1)
            #     v_net_input_std = self.agent.ac.v_net.input_layer.weight.data.permute(1, 0).reshape(12, -1).std(dim=1)
            # with open(f'{cfg.OUT_DIR}/weight_stats/input_{cur_iter}.pkl', 'wb') as f:
            #     stats = {
            #         'mu_net': {'mean': mu_net_input_mean.cpu(), 'std': mu_net_input_std.cpu()}, 
            #         'v_net': {'mean': v_net_input_mean.cpu(), 'std': v_net_input_std.cpu()}
            #     }
            #     pickle.dump(stats, f)
            # if cfg.MODEL.MLP.HN_INPUT:
            #     stats = {
            #         'mu_net': mu_context_embedding, 
            #         'v_net': v_context_embedding, 
            #     }
            #     with open(f'{cfg.OUT_DIR}/weight_stats/input_context_embedding_{cur_iter}.pkl', 'wb') as f:
            #         pickle.dump(stats, f)
            #     stats = {
            #         'mean': obs_mean.cpu() / cfg.PPO.TIMESTEPS, 'std': obs_std.cpu() / cfg.PPO.TIMESTEPS
            #     }
            #     with open(f'{cfg.OUT_DIR}/weight_stats/obs_{cur_iter}.pkl', 'wb') as f:
            #         pickle.dump(stats, f)

            # if not cfg.MODEL.MLP.HN_OUTPUT:
            #     mu_net_output_mean = self.agent.ac.mu_net.output_layer.weight.data.reshape(12, -1).mean(dim=1)
            #     mu_net_output_std = self.agent.ac.mu_net.output_layer.weight.data.reshape(12, -1).std(dim=1)
            #     v_net_output_mean = self.agent.ac.v_net.output_layer.weight.data.reshape(12, -1).mean(dim=1)
            #     v_net_output_std = self.agent.ac.v_net.output_layer.weight.data.reshape(12, -1).std(dim=1)
            # with open(f'{cfg.OUT_DIR}/weight_stats/output_{cur_iter}.pkl', 'wb') as f:
            #     stats = {
            #         'mu_net': {'mean': mu_net_output_mean, 'std': mu_net_output_std}, 
            #         'v_net': {'mean': v_net_output_mean, 'std': v_net_output_std}
            #     }
            #     pickle.dump(stats, f)

            self.train_meter.update_iter_returns(cur_iter)

            unimal_ids = self.envs.get_unimal_idx()
            next_val = self.agent.get_value(obs, unimal_ids=unimal_ids)
            self.buffer.compute_returns(next_val)
            # filter out unstable trajectories
            self.buffer.filter(process_with_error)
            # train on batch
            self.train_on_batch(cur_iter)

            # with open(f'{cfg.OUT_DIR}/iter_sampled_agents/{cur_iter}.pkl', 'wb') as f:
            #     pickle.dump(self.sampled_agents, f)

            if cfg.ENV.TASK_SAMPLING == 'UED' and cfg.UED.CURATION in ['positive_value_loss', 'L1_value_loss', 'GAE']:
                # update UED scores of the agents sampled in the current iteration
                self.task_sampler.update_scores(self.buffer)
                with open(f'{cfg.OUT_DIR}/proxy_score/{cur_iter}.pkl', 'wb') as f:
                    pickle.dump([self.task_sampler.potential_score, self.task_sampler.staleness_score], f)

            # self.train_meter.update_mean()
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
                self._log_fps(cur_iter, log=False)
                stats = self.train_meter.get_stats()
                stats["fps"] = self.fps
                fu.save_json(stats, path)
                print (cfg.OUT_DIR)

            self._log_fps(cur_iter, log=True)

            if (cur_iter + 1) % model_save_freq == 0:
                self.save_model(cur_iter + 1)
            
            if cfg.UED.USE_VALIDATION:

                # evaluate on the validation robots and move the converged ones to the training set
                if cur_iter % cfg.UED.CHECK_VALID_FREQ == 0 and cur_iter >= cfg.UED.VALIDATION_START_ITER:
                    print ('evaluate on the validation set')
                    set_ob_rms(self.valid_envs, get_ob_rms(self.envs))
                    # TODO (low priority): the reset is performed based on the chunk in the previous iter
                    self.valid_envs.reset()
                    for step in range(cfg.UED.VALID_TIMESTEPS):
                        _, act, _, _, _ = self.agent.act(obs, compute_val=False)
                        obs, reward, done, infos = self.valid_envs.step(act)
                        self.valid_meter.add_ep_info(infos, cur_iter, np.where(done)[0])
                    self.valid_meter.update_iter_returns(cur_iter)
                    # with Pool(processes=16) as pool:
                    #     valid_returns = pool.map(eval_validation, self.valid_meter.agents)
                    # for agent in self.valid_meter.agents:
                    #     episode_return = eval_validation(agent)
                    #     self.valid_meter.agent_meters[agent].add_new_score(cur_iter, episode_return, 8)
                    # for agent, score in zip(self.valid_meter.agents, valid_returns):
                    #     self.valid_meter.agent_meters[agent].add_new_score(cur_iter, score, 8)
                    
                    # find converged valid agents
                    agents_to_move = []
                    print ('move the following agents to the training set')
                    for agent in self.valid_meter.agents:
                        agent_meter = self.valid_meter.agent_meters[agent]
                        if len(agent_meter.iter_mean_return) == 0:
                            continue
                        # TODO: as we have more validation agents, the return estimation may be less accurate as evaluation time on each robot is less
                        # TODO: can we maintain a fixed-size training and validation pool?
                        if agent_meter.iter_mean_return[-1] < agent_meter.best_iter_return and agent_meter.iter_idx[-1] - agent_meter.best_iter >= cfg.UED.VALID_MAX_WAIT * cfg.UED.CHECK_VALID_FREQ:
                            print (agent, agent_meter.iter_mean_return, agent_meter.best_iter_return)
                            agents_to_move.append(agent)
                    # move the converged validation agents to the training set
                    self.valid_meter.delete_agents(agents_to_move)
                    self.train_meter.add_new_agents(agents_to_move, cur_iter=cur_iter)
                    cfg.ENV.WALKERS.extend(agents_to_move)
                    # mutate the moved agents to get new validation agents
                    # TODO: we can learn where to generate new validation agents
                    print ('add new agents to the validation set')
                    new_valid_agents = []
                    if len(agents_to_move) > 0:
                        process_num = min(50, len(agents_to_move))
                        with Pool(processes=process_num) as pool:
                            mutated_valid_agents = pool.starmap(mutate_robot, [(agent, cfg.ENV.WALKER_DIR) for agent in agents_to_move])
                        for new_agents in mutated_valid_agents:
                            new_valid_agents.extend(new_agents)
                    # add new randomly sampled validation agents
                    random_agent_id = [cur_iter // cfg.UED.CHECK_VALID_FREQ * cfg.UED.RANDOM_VALIDATION_ROBOT_NUM + idx for idx in range(cfg.UED.RANDOM_VALIDATION_ROBOT_NUM)]
                    with Pool(processes=cfg.UED.RANDOM_VALIDATION_ROBOT_NUM) as pool:
                        random_valid_agents = pool.starmap(sample_new_robot, [(f'random_{agent_id}', cfg.ENV.WALKER_DIR) for agent_id in random_agent_id])
                    new_valid_agents.extend(random_valid_agents)
                    # for idx in range(cfg.UED.RANDOM_VALIDATION_ROBOT_NUM):
                    #     agent_id = cur_iter // cfg.UED.CHECK_VALID_FREQ * cfg.UED.RANDOM_VALIDATION_ROBOT_NUM + idx
                    #     agent = sample_new_robot(f'random_{agent_id}', cfg.ENV.WALKER_DIR)
                    #     new_valid_agents.append(agent)

                    # update the valid meter
                    self.valid_meter.add_new_agents(new_valid_agents, cur_iter=cur_iter)
                    # save the updated validation set
                    with open(f'{cfg.OUT_DIR}/walkers_valid.json', 'w') as f:
                        json.dump(self.valid_meter.agents, f)
                    self.save_valid_sampled_agent_seq()

                # Do not need to this if we curate the training agents via learning progress
                # remove training robots with no performance improvement
                # if cur_iter % cfg.UED.CHECK_TRAIN_FREQ == 0:
                #     uncontrollable_agents = []
                #     for agent in self.train_meter.agents:
                #         agent_meter = self.train_meter.agent_meters[agent]
                #         # check performance convergencce
                #         if len(agent_meter.iter_mean_return) == 0:
                #             continue
                #         # TODO: learning progress also depends on how often the robot is sampled for training
                #         if agent_meter.iter_mean_return[-1] < agent_meter.best_iter_return and cur_iter - agent_meter.best_iter >= 5 * cfg.UED.CHECK_TRAIN_FREQ:
                #             # TODO: this is not reasonable at the early stage of training
                #             if agent_meter.best_iter_return < 500:
                #                 uncontrollable_agents.append(agent)
                #     print ('remove the following agents from the training set')
                #     print (uncontrollable_agents)
                #     self.train_meter.delete_agents(uncontrollable_agents)
                #     for agent in uncontrollable_agents:
                #         cfg.ENV.WALKERS.remove(agent)

            # generate new agents by mutating existing ones
            if cfg.ENV.TASK_SAMPLING == 'UED' and cfg.UED.GENERATION:
                if cur_iter >= 30 and cur_iter % cfg.UED.GENERATION_FREQ == 0:
                    agents = self.train_meter.agents
                    if cfg.UED.PARENT_SELECT_STRATEGY == 'learning_progress':
                        score = [self.train_meter.agent_meters[agent].get_learning_speed() for agent in agents]
                        for i in range(len(score)):
                            if score[i] < 0:
                                score[i] = 1e-4
                        # filter out robots that do not make significant learning progress
                        # candidate_idx = np.where(np.array(score) >= 10)[0]
                        # # filter out robots that have been mutated many times
                        # final_candidate_idx = []
                        # for idx in candidate_idx:
                        #     agent = agents[idx]
                        #     if self.train_meter.agent_meters[agent].children_num < 5:
                        #         final_candidate_idx.append(idx)
                        final_candidate_idx = [i for i, agent in enumerate(agents) if self.train_meter.agent_meters[agent].children_num < 5]
                    elif cfg.UED.PARENT_SELECT_STRATEGY == 'uniform':
                        score = np.ones(len(agents))
                        final_candidate_idx = [i for i, agent in enumerate(agents) if self.train_meter.agent_meters[agent].children_num < 5]

                    if len(final_candidate_idx) > 0:
                        # sample agents for mutation
                        if len(final_candidate_idx) <= cfg.UED.GENERATION_NUM:
                            parents = [agents[idx] for idx in final_candidate_idx]
                        else:
                            probs = np.array([score[idx] for idx in final_candidate_idx])
                            # make the sampling more balanced
                            if cfg.UED.BALANCE_GENERATION:
                                penalty_score = []
                                for idx in final_candidate_idx:
                                    agent = agents[idx]
                                    parts = agent.split('-mutate-')
                                    if len(parts) > 1 and len(parts[1]) == 1:
                                        root = parts[0] + '-mutate-' + parts[1]
                                    else:
                                        root = parts[0]
                                    penalty_score.append(self.train_meter.agent_meters[root].tree_size + 1)
                                penalty_score = np.array(penalty_score)
                                probs = probs / penalty_score
                            # sample based on the final prob
                            probs = probs / probs.sum()
                            idx = np.random.choice(final_candidate_idx, cfg.UED.GENERATION_NUM, p=probs, replace=False)
                            parents = [agents[i] for i in idx]
                        # update the children number and tree size
                        for parent in parents:
                            self.train_meter.agent_meters[parent].children_num += 1
                            node = parent
                            while node:
                                self.train_meter.agent_meters[node].tree_size += 1
                                node = self.train_meter.agent_meters[node].parent
                        print ('mutate the following agents: ')
                        for i, parent in zip(idx, parents):
                            print (parent, f'score: {score[i]}')
                        # mutate the parents
                        mutated_agents, mutation_parents = [], []
                        for parent in parents:
                            mutate_id = self.train_meter.agent_meters[parent].children_num
                            new_agent = mutate_single_robot(parent, cfg.ENV.WALKER_DIR, mutate_id, grow_limb_only=cfg.UED.GROW_LIMB_ONLY)
                            if new_agent is not None:
                                mutated_agents.append(new_agent)
                                mutation_parents.append(parent)
                        # add the new agents to the agent list and meter
                        cfg.ENV.WALKERS.extend(mutated_agents)
                        self.train_meter.add_new_agents(mutated_agents, cur_iter=cur_iter)
                        for agent, parent in zip(mutated_agents, mutation_parents):
                            self.train_meter.agent_meters[agent].parent = parent
                        print ('get the following mutated agents: ')
                        for agent in mutated_agents:
                            print (agent)

            # if cur_iter >= 0 and cur_iter % cfg.UED.ADD_NEW_AGENTS_FREQ == 0:
            #     agent_scores = []
            #     for agent in cfg.ENV.WALKERS:
            #         scores = self.train_meter.agent_meters[agent].mean_ep_rews['reward']
            #         if len(scores) < 5:
            #             agent_scores.append(0.)
            #         else:
            #             agent_scores.append(np.array(scores[-5:]).mean())
            #     new_agents, new_agents_parent = self.agent_manager.generate_new_agents(set(self.sampled_agents), np.array(agent_scores))
            #     if len(new_agents) > 0:
            #         new_potential_score = [np.mean(self.train_meter.agent_meters[agent].ep_positive_gae) for agent in new_agents_parent]
            #         self.agent_manager.initialize_new_agents_score(new_agents, new_potential_score)
            #         cfg.ENV.WALKERS.extend(new_agents)
            #         num_agents = len(cfg.ENV.WALKERS)
            #         print (f'The agent pool now contains {num_agents} agents')
            #         with open(f'{cfg.OUT_DIR}/walkers.json', 'w') as f:
            #             json.dump(cfg.ENV.WALKERS, f)
            #         self.train_meter.add_new_agents(new_agents)
            #     else:
            #         print ('no new agents added in this iteration')

            # remove unstable robots from the walkers
            robot_with_error = set(robot_with_error)
            self.train_meter.delete_agents(robot_with_error)
            for agent in robot_with_error:
                cfg.ENV.WALKERS.remove(agent)
            with open(f'{cfg.OUT_DIR}/walkers_train.json', 'w') as f:
                json.dump(cfg.ENV.WALKERS, f)
            # sample agents for next iteration
            self.save_sampled_agent_seq(cur_iter)

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret[:, self.buffer.idx] - self.buffer.val[:, self.buffer.idx]
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        ratio_hist = [[] for _ in range(cfg.PPO.EPOCHS)]
        grad_norm_dict, grad_correlation_dict = defaultdict(list), defaultdict(list)
        limb_ratio_record = []

        for i in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for j, batch in enumerate(batch_sampler):
                # Reshape to do in a single forward pass for all steps
                val, _, logp, ent = self.actor_critic(batch["obs"], batch["act"], \
                    # dropout_mask_v=batch['dropout_mask_v'], \
                    # dropout_mask_mu=batch['dropout_mask_mu'], \
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

        # self.std_record.append(std_record)

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
        torch.save([self.actor_critic.state_dict(), get_ob_rms(self.envs), get_ret_rms(self.envs), self.optimizer.state_dict()], path)
        checkpoint_path = os.path.join(cfg.OUT_DIR, f"checkpoint_{cur_iter}.pt")
        torch.save([self.actor_critic.state_dict(), get_ob_rms(self.envs), get_ret_rms(self.envs), self.optimizer.state_dict()], checkpoint_path)

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
            
            _, act, _ = self.agent.act(obs)
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

            if cfg.UED.USE_VALIDATION:
                if cur_iter >= cfg.UED.VALIDATION_START_ITER:
                    # assign more weights to agents that are less sampled (especially the newly added ones)
                    staleness_score = np.exp(-np.array([self.train_meter.agent_meters[agent].discounted_total_episode_num for agent in agents]))
                    staleness_score = staleness_score / staleness_score.sum()
                    # learning progress score
                    potential_score = np.array([self.train_meter.agent_meters[agent].get_learning_speed() for agent in agents])
                    potential_score[potential_score == -1] = potential_score.max()
                    potential_score = potential_score / potential_score.sum()

                    probs = potential_score * (1. - cfg.UED.STALENESS_WEIGHT) + staleness_score * cfg.UED.STALENESS_WEIGHT

                    with open(f'{cfg.OUT_DIR}/proxy_score/{cur_iter}.pkl', 'wb') as f:
                        pickle.dump({'LP': potential_score, 'staleness': staleness_score}, f)

                    # ep_count = np.array([self.train_meter.agent_meters[agent].ep_count for agent in agents])
                    # probs = ep_count.max() - ep_count
                    # probs = probs / probs.sum()
                    # probs = probs * 0.8 + np.ones(num_agents) / num_agents * 0.2
                else:
                    probs = [1. for _ in agents]
            
            elif cfg.UED.CURATION in ['positive_value_loss', 'L1_value_loss', 'GAE']:
                # select new agents for the next learning iteration
                # TODO: adaptively change this threshold based on value loss?
                if cur_iter >= int(cfg.PPO.MAX_ITERS / 20.):
                    print (f'curation with {cfg.UED.CURATION}')
                    probs = self.task_sampler.get_sample_probs(cur_iter)
                else:
                    probs = [1. for _ in agents]
            
            elif cfg.UED.CURATION == 'regret':

                if self.ST_performance is None:
                    # hardcode it here only for proof-of-concept experiments
                    with open(cfg.UED.UPPER_BOUND_PATH, 'r') as f:
                        self.ST_performance = json.load(f)

                ep_lens = [len(self.train_meter.agent_meters[agent].ep_len) for agent in agents]
                if min(ep_lens) == 10:
                    print ('curation with true regret')
                    # TODO: how to deal with negative regret
                    # Issue: training score is computed based on an out-of-date model, 
                    # which may not correctly reflect the current model's performance on a robot
                    if cfg.UED.REGRET_TYPE == 'absolute':
                        regret = [
                            self.ST_performance[agent] - np.mean(self.train_meter.agent_meters[agent].ep_rew["reward"]) \
                            for agent in agents
                        ]
                        regret = np.maximum(regret, 0.)
                    elif cfg.UED.REGRET_TYPE == 'relative':
                        regret = [
                            1. - np.mean(self.train_meter.agent_meters[agent].ep_rew["reward"]) / self.ST_performance[agent] \
                            for agent in agents
                        ]
                        regret = np.clip(regret, 0., 1.)
                    regret = regret / regret.sum()
                    probs = regret * 0.9 + np.ones(len(regret)) / len(regret) * 0.1
                else:
                    ep_lens = [l + 1 for l in ep_lens]
                    probs = [1000.0 / l for l in ep_lens]
            
            elif cfg.UED.CURATION == 'learning_progress':
                # ep_lens = [len(self.train_meter.agent_meters[agent].ep_len) for agent in agents]
                if cur_iter < 30:
                    probs = [1. for _ in agents]
                else:
                # elif min(ep_lens) == 10:
                    print ('start to sample based on learning progress')
                    LP_score = np.array([self.train_meter.agent_meters[agent].get_learning_speed() for agent in agents])
                    print (np.sort(LP_score))
                    with open(f'{cfg.OUT_DIR}/proxy_score/{cur_iter}.pkl', 'wb') as f:
                        pickle.dump({'LP': LP_score}, f)
                    LP_score[LP_score == -1] = LP_score.max()

                    probs = LP_score / LP_score.sum()
                    probs = probs * 0.9 + np.ones(len(probs)) / len(probs) * 0.1
                # else:
                #     ep_lens = [l + 1 for l in ep_lens]
                #     probs = [1000.0 / l for l in ep_lens]

            else:
                probs = [1. for _ in agents]

        # maybe use softmax here?
        # probs = np.exp(probs)
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = probs / probs.sum()

        # smooth the change of sampling prob
        if cfg.UED.PROB_CHANGE_RATE is not None:
            if cur_iter == -1:
                self.last_probs = probs
            else:
                # TODO: the logic here is not correct
                if self.last_probs is None:
                    last_iter = int(cfg.PPO.CHECKPOINT_PATH[:-3].split('_')[-1]) - 1
                    with open(f'{cfg.OUT_DIR}/iter_prob/{last_iter}.pkl', 'rb') as f:
                        self.last_probs = pickle.load(f)
                if len(self.last_probs) != len(probs):
                    self.last_probs = np.concatenate([self.last_probs, np.zeros(len(probs) - len(self.last_probs))])
                delta_prob = probs - self.last_probs
                probs = self.last_probs + cfg.UED.PROB_CHANGE_RATE * delta_prob
                self.last_probs = probs = probs / probs.sum()

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