import gym
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from metamorph.config import cfg


class Buffer(object):
    def __init__(self, obs_space, act_shape):
        N, T, P = cfg.DAGGER.ITERS, cfg.DAGGER.ITER_STEPS, cfg.PPO.NUM_ENVS
        self.P = P

        if isinstance(obs_space, gym.spaces.Dict):
            self.obs = {}
            for obs_type, obs_space_ in obs_space.spaces.items():
                self.obs[obs_type] = torch.zeros(N*T, P, *obs_space_.shape)
        else:
            self.obs = torch.zeros(N*T, P, *obs_space.shape)

        self.act_target = torch.zeros(N*T, P, *act_shape)

        self.step = 0

    def to(self, device):
        if isinstance(self.obs, dict):
            for obs_type, obs_space in self.obs.items():
                self.obs[obs_type] = self.obs[obs_type].to(device)
        else:
            self.obs = self.obs.to(device)
        self.act_target = self.act_target.to(device)

    def insert(self, obs, act_target):
        if isinstance(obs, dict):
            for obs_type, obs_val in obs.items():
                self.obs[obs_type][self.step] = obs_val.cpu()
        else:
            self.obs[self.step] = obs.cpu()
        self.act_target[self.step] = act_target.cpu()

        # self.step = (self.step + 1) % cfg.DAGGER.ITER_STEPS
        self.step += 1

    def get_sampler(self):

        dset_size = self.step * self.P
        assert dset_size >= cfg.DAGGER.BATCH_SIZE

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            cfg.DAGGER.BATCH_SIZE,
            drop_last=True,
        )

        for idxs in sampler:
            batch = {}
            if isinstance(self.obs, dict):
                batch["obs"] = {}
                for ot, ov in self.obs.items():
                        batch["obs"][ot] = ov[:self.step].view(-1, *ov.size()[2:])[idxs].cuda()
            else:
                batch["obs"] = self.obs[:self.step].view(-1, *self.obs.size()[2:])[idxs].cuda()

            batch["act_target"] = self.act_target[:self.step].view(-1, self.act_target.size(-1))[idxs].cuda()
            yield batch
