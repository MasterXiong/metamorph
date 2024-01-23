import os
import copy
import json
import pickle
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metamorph.config import cfg, get_default_cfg
from metamorph.algos.ppo.model import ActorCritic
from metamorph.algos.ppo.inherit_weight import restore_from_checkpoint
from metamorph.algos.ppo.envs import *
from metamorph.algos.distill.buffer import Buffer
from metamorph.utils import file as fu

torch.manual_seed(0)
np.random.seed(0)


def convert_dict_to_list(config_dict, prefix):
    config_list = []

    def traverse(configs, level_name):
        for key, value in configs.items():
            if type(value) == dict:
                traverse(value, '.'.join([level_name, key]))
            else:
                config_list.extend(['.'.join([level_name, key]), value])
    
    traverse(config_dict, prefix)
    return config_list


class DAggerTrainer:
    def __init__(self):
        self.device = torch.device(cfg.DEVICE)

        with open(f'{cfg.DAGGER.TEACHER_PATH}/config.yaml', 'r') as f:
            teacher_config = yaml.safe_load(f)
        teacher_model_config = convert_dict_to_list(teacher_config['MODEL'], 'MODEL')
        for i in range(len(teacher_model_config) // 2):
            try:
                cfg.merge_from_list(teacher_model_config[(i*2):(i+1)*2])
            except:
                continue
        # initialize the envs with the teacher's observation types
        # hardcode here, as in the code we can only rebuild the student's obs with the teacher's obs, not vice versa
        self.envs = make_vec_envs(training=False)
        # load the teacher model
        self.teacher = ActorCritic(self.envs.observation_space, self.envs.action_space).cuda()
        # use the teacher's obs normalizer
        obs_rms, _, _ = restore_from_checkpoint(self.teacher, cp_path=f'{cfg.DAGGER.TEACHER_PATH}/Unimal-v0.pt')
        set_ob_rms(self.envs, obs_rms)

        with open(f'{cfg.DAGGER.STUDENT_PATH}/config.yaml', 'r') as f:
            student_config = yaml.safe_load(f)
        student_model_config = convert_dict_to_list(student_config['MODEL'], 'MODEL')
        for i in range(len(student_model_config) // 2):
            try:
                cfg.merge_from_list(student_model_config[(i*2):(i+1)*2])
            except:
                continue
        # student envs is only used to set the state and action space of the student model
        student_envs = make_vec_envs(training=False)
        # load the model
        self.student = ActorCritic(student_envs.observation_space, student_envs.action_space).cuda()
        # the student and teacher share the same obs_rms based on our distillation code
        self.student_obs_rms, _, _ = restore_from_checkpoint(self.student, cp_path=f'{cfg.DAGGER.STUDENT_PATH}/checkpoint_{cfg.DAGGER.STUDENT_CHECKPOINT}.pt')
        # buffer
        self.buffer = Buffer(student_envs.observation_space, student_envs.action_space.shape)

        # TODO: student may have different context features
        if os.path.exists(f'{cfg.ENV.WALKER_DIR}/context_v2.pkl'):
            with open(f'{cfg.ENV.WALKER_DIR}/context_v2.pkl', 'rb') as f:
                self.student_context = pickle.load(f)
        else:
            self.student_context = []
            for agent in cfg.ENV.WALKERS:
                env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
                init_obs = env.reset()
                self.student_context.append(init_obs['context'])
                env.close()
            self.student_context = np.stack(self.student_context)
            self.student_context = torch.from_numpy(self.student_context).float()
            with open(f'{cfg.ENV.WALKER_DIR}/context_v2.pkl', 'wb') as f:
                pickle.dump(self.student_context, f)

        self.optimizer = optim.Adam(self.student.parameters(), lr=cfg.DAGGER.BASE_LR, eps=cfg.DAGGER.EPS, weight_decay=cfg.DAGGER.WEIGHT_DECAY)

    def convert_obs(self, teacher_obs, unimal_ids):
        student_obs = copy.deepcopy(teacher_obs)
        student_obs['context'] = self.student_context[unimal_ids].cuda()
        if 'HN-MLP' in cfg.DAGGER.STUDENT_PATH:
            dims = list(range(13)) + [30, 31] + [41, 42]
            data_size = student_obs['proprioceptive'].shape[0]
            new_obs = student_obs['proprioceptive'].view(data_size, cfg.MODEL.MAX_LIMBS, -1)
            new_obs = new_obs[:, :, dims]
            student_obs['proprioceptive'] = new_obs.view(data_size, -1)
        return student_obs

    def train(self):

        with open(f'{cfg.OUT_DIR}/walkers_train.json', 'w') as f:
            json.dump(cfg.ENV.WALKERS, f)
        self.save_sampled_agent_seq(-1)

        obs = self.envs.reset()
        for cur_iter in range(cfg.DAGGER.ITERS):
            for step in range(cfg.DAGGER.ITER_STEPS):
                unimal_ids = self.envs.get_unimal_idx()
                # teacher
                with torch.no_grad():
                    _, teacher_pi, _, _ = self.teacher(obs, compute_val=False)
                teacher_act_mean = teacher_pi.loc
                teacher_act_sample = teacher_pi.sample()
                # student
                student_obs = self.convert_obs(obs, unimal_ids)
                with torch.no_grad():
                    _, student_pi, _, _ = self.student(student_obs, compute_val=False)
                student_act_mean = student_pi.loc
                student_act_sample = student_pi.sample()
                # Sample actions
                anneal_threshold = 1. - cur_iter / cfg.DAGGER.ITERS
                if cur_iter < 10:
                    anneal_threshold = 1. - cur_iter / 10
                    choose_teacher = (torch.rand(teacher_act_mean.shape[0], 1) <= anneal_threshold).float().cuda()
                    act = choose_teacher * teacher_act_sample + (1. - choose_teacher) * student_act_sample
                else:
                    act = student_act_sample
                # save to buffer
                self.buffer.insert(student_obs, teacher_act_mean)
                # env step
                obs, reward, done, infos = self.envs.step(act)
            
            print (f'Train for iter {cur_iter}')
            self.train_on_buffer()

            self.save_sampled_agent_seq(cur_iter)
            if (cur_iter + 1) % cfg.DAGGER.SAVE_FREQ == 0:
                torch.save([self.student.state_dict(), self.student_obs_rms], f'{cfg.OUT_DIR}/checkpoint_{cur_iter + 1}.pt')

    def train_on_buffer(self):

        update_num = 0
        early_stop = False
        batch_losses = []
        for i in range(cfg.DAGGER.EPOCH_PER_ITER):
            batch_sampler = self.buffer.get_sampler()
            for j, batch in enumerate(batch_sampler):
                _, student_pi, _, _ = self.student(batch["obs"], compute_val=False)
                student_act_mean = student_pi.loc
                # compute action loss
                loss = (((student_act_mean - batch['act_target']).square() * (1 - batch["obs"]['act_padding_mask'].float())).sum(dim=-1) / (1 - batch["obs"]['act_padding_mask'].float()).sum(dim=1)).mean()
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.student.parameters(), 0.5)
                self.optimizer.step()
                update_num += 1
                if update_num % 10 == 0:
                    print (f'avg loss over last 10 batches: {np.mean(batch_losses[-10:])}')
                if update_num == cfg.DAGGER.MINIBATCH_UPDATE_PER_ITER:
                    early_stop = True
                    break

            if early_stop:
                break

    def save_sampled_agent_seq(self, cur_iter):
        
        agents = cfg.ENV.WALKERS
        num_agents = len(agents)
        if num_agents <= 1:
            return

        ep_lens = [1000] * num_agents
        probs = [1000.0 / l for l in ep_lens]

        # probs = np.exp(probs)
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = probs / probs.sum()

        ep_per_env = 10
        # Task list size (multiply by 8 as padding)
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, f"sampling_train.json")
        fu.save_json(task_list, path)
