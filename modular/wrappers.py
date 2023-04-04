import os
import random

import gym
import numpy as np
from gym import spaces
from gym import utils

from gym.spaces import Box
from gym.spaces import Dict

from metamorph.config import cfg
from metamorph.envs.modules.agent import create_agent_xml
from metamorph.envs.tasks.unimal import UnimalEnv
from metamorph.utils import file as fu
from metamorph.utils import spaces as spu


class ModularObservationPadding(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        num_limbs = self.metadata["num_limbs"]
        self.num_limb_pads = self.max_limbs - num_limbs

        self.limb_obs_size = self.observation_space.shape[0] // num_limbs

        inf = np.float32(np.inf)
        self.observation_space = dict()
        shape = (self.limb_obs_size * cfg.MODEL.MAX_LIMBS,)
        self.observation_space['proprioceptive'] = Box(-inf, inf, shape, np.float32)
        self.observation_space['context'] = Box(-inf, inf, shape, np.float32)
        self.observation_space['obs_padding_mask'] = Box(-inf, inf, (self.max_limbs, ), np.float32)
        self.observation_space['act_padding_mask'] = Box(-inf, inf, (self.max_limbs, ), np.float32)
        self.observation_space['edges'] = Box(-inf, inf, (self.max_joints * 2, ), np.float32)
        self.observation_space = Dict(self.observation_space)

        obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads
        self.obs_padding_mask = np.asarray(obs_padding_mask)

        act_padding_mask = [True] + [False] * (num_limbs - 1) + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def observation(self, obs):

        padding = [0.0] * (self.limb_obs_size * self.num_limb_pads)
        obs_dict = dict()

        obs_dict["proprioceptive"] = np.concatenate([obs, padding]).ravel()
        obs_dict["context"] = np.concatenate([obs, padding]).ravel()
        obs_dict["obs_padding_mask"] = self.obs_padding_mask
        obs_dict["act_padding_mask"] = self.act_padding_mask
        obs_dict["edges"] = np.zeros(self.max_joints * 2)

        return obs_dict

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


class ModularActionPadding(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_LIMBS
        self._update_action_space()
        act_padding_mask = [True] + [False] * self.metadata["num_joints"] + [True] * self.num_limb_pads
        self.act_padding_mask = np.asarray(act_padding_mask)

    def _update_action_space(self):
        num_pads = self.max_limbs - self.metadata["num_limbs"] - 1
        low, high = self.action_space.low, self.action_space.high
        low = np.concatenate([[-1.], low, [-1.] * num_pads]).astype(np.float32)
        high = np.concatenate([[-1.], high, [-1.] * num_pads]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        new_action = action[~self.act_padding_mask]
        return new_action
