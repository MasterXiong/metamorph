import gym
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from metamorph.config import cfg


class Buffer(object):
    def __init__(self, obs_space, act_shape):
        T, P = cfg.DAGGER.ITER_STEPS, cfg.PPO.NUM_ENVS

        if isinstance(obs_space, gym.spaces.Dict):
            self.obs = {}
            for obs_type, obs_space_ in obs_space.spaces.items():
                self.obs[obs_type] = torch.zeros(T, P, *obs_space_.shape)
        else:
            self.obs = torch.zeros(T, P, *obs_space.shape)

        self.act_mean = torch.zeros(T, P, *act_shape)

        self.step = 0

    def to(self, device):
        if isinstance(self.obs, dict):
            for obs_type, obs_space in self.obs.items():
                self.obs[obs_type] = self.obs[obs_type].to(device)
        else:
            self.obs = self.obs.to(device)
        self.act_mean = self.act_mean.to(device)

    def insert(self, obs, act_mean):
        if isinstance(obs, dict):
            for obs_type, obs_val in obs.items():
                self.obs[obs_type][self.step] = obs_val
        else:
            self.obs[self.step] = obs
        self.act_mean[self.step] = act_mean

        self.step = (self.step + 1) % DAGGER.ITER_STEPS

    def get_sampler(self, adv):
        dset_size = cfg.PPO.TIMESTEPS * adv.shape[1]

        assert dset_size >= cfg.PPO.BATCH_SIZE

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            cfg.PPO.BATCH_SIZE,
            drop_last=True,
        )

        for idxs in sampler:
            batch = {}
            batch["ret"] = self.ret[:, self.idx].view(-1, 1)[idxs]

            if isinstance(self.obs, dict):
                batch["obs"] = {}
                for ot, ov in self.obs.items():
                        batch["obs"][ot] = ov[:, self.idx].view(-1, *ov.size()[2:])[idxs]
            else:
                batch["obs"] = self.obs[:, self.idx].view(-1, *self.obs.size()[2:])[idxs]

            batch["val"] = self.val[:, self.idx].view(-1, 1)[idxs]
            batch["act"] = self.act[:, self.idx].view(-1, self.act.size(-1))[idxs]
            batch["adv"] = adv.view(-1, 1)[idxs]
            batch["logp_old"] = self.logp[:, self.idx].view(-1, 1)[idxs]
            # batch["dropout_mask_v"] = self.dropout_mask_v.view(-1, 12, 128)[idxs]
            # batch["dropout_mask_mu"] = self.dropout_mask_mu.view(-1, 12, 128)[idxs]
            batch["unimal_ids"] = self.unimal_ids[:, self.idx].view(-1)[idxs]
            # batch["limb_logp_old"] = self.limb_logp.view(-1, 24)[idxs]
            yield batch

    def filter(self, process_with_error):
        self.idx = [True for _ in range(cfg.PPO.NUM_ENVS)]
        for p in process_with_error:
            self.idx[p] = False
