import pickle
import json
import numpy as np
import copy
from collections import defaultdict
import torch

from metamorph.config import cfg

from derl.envs.morphology import SymmetricUnimal
from derl.utils import xml as xu


def find_removable_limbs(agent):
    limb_list = agent.limb_list.copy()
    removable_limbs = []
    for limbs in limb_list:
        body = xu.find_elem(agent.unimal, "body", "name", "limb/{}".format(limbs[0]))[0]
        num_children = len(xu.find_elem(body, "body", child_only=True))
        if num_children == 0:
            removable_limbs.append(limbs)
    return removable_limbs


def mutate_robot(agent, folder):
    mutated_agents = []
    # generate all possible mutations by removing limb(s)
    parent_unimal = SymmetricUnimal(agent, f'{folder}/unimal_init/{agent}.pkl')
    limbs = find_removable_limbs(parent_unimal)
    for i, l in enumerate(limbs):
        if parent_unimal.num_limbs <= 3:
            break
        name = f'{agent}-mutate-remove-{i}'
        unimal = SymmetricUnimal(name, f'{folder}/unimal_init/{agent}.pkl')
        unimal.mutate_delete_limb(limb_to_remove=l)
        if unimal.num_limbs <= 2:
            continue
        unimal.save(folder)
        pickle_to_json(folder, unimal.id)
        mutated_agents.append(name)

    # add limbs
    add_limb_mutation_num = 5 - len(mutated_agents)
    i = 0
    for i in range(add_limb_mutation_num):
        new_agent = f'{agent}-mutate-add-{i}'
        unimal = SymmetricUnimal(new_agent, f'{folder}/unimal_init/{agent}.pkl')
        unimal.grow_limb()
        if unimal.num_limbs == parent_unimal.num_limbs:
            break
        if unimal.num_limbs >= 12 or len(xu.find_elem(unimal.actuator, "motor")) > 16:
            continue
        unimal.save(folder)
        pickle_to_json(folder, unimal.id)
        mutated_agents.append(new_agent)

    for op in ["limb_params", "gear", "dof", "joint_angle", "density"]:
        new_agent = f'{agent}-mutate-{op}'
        unimal = SymmetricUnimal(new_agent, f'{folder}/unimal_init/{agent}.pkl')
        unimal.mutate(op=op)
        if unimal.num_limbs >= 12 or len(xu.find_elem(unimal.actuator, "motor")) > 16:
            continue
        if unimal.num_limbs <= 2:
            continue
        unimal.save(folder)
        pickle_to_json(folder, unimal.id)
        mutated_agents.append(new_agent)
    
    return mutated_agents

# def rollout(policy, env, agent, num_envs=16):
#     episode_return = np.zeros(num_envs)
#     not_done = np.ones(num_envs)

#     obs = env.reset()
#     for t in range(2000):
#         val, act, _, _, _ = policy.act(obs, return_attention=False)
#         obs, reward, done, infos = env.step(act)
#         idx = np.where(done)[0]
#         for i in idx:
#             if not_done[i] == 1:
#                 not_done[i] = 0
#                 episode_return[i] = infos[i]['episode']['r']
#         if not_done.sum() == 0:
#             break


def pickle_to_json(folder, name):
    with open(f'{folder}/unimal_init/{name}.pkl', 'rb') as f:
        states = pickle.load(f)
    # for key in states:
    #     print (key)
    #     print (states[key])
    metadata = {}
    metadata['dof'] = states['dof']
    metadata['num_limbs'] = states['num_limbs']
    metadata['symmetric_limbs'] = [limbs for limbs in states['limb_list'] if len(limbs) == 2]
    with open(f'{folder}/metadata/{name}.json', 'w') as f:
        json.dump(metadata, f)


class TaskSampler:

    def __init__(
        self, 
        init_agents, 
        staleness_score_weight=0.1, 
        mutation_agent_num=100, 
        potential_score_EMA_coef=0.5, 
    ):
        # TODO: maybe need to use some optimistic intialization for potential score
        self.agents = copy.deepcopy(init_agents)
        self.potential_score = np.zeros(len(init_agents))
        self.staleness_score = np.zeros(len(init_agents))
        self.staleness_score_weight = staleness_score_weight
        self.mutation_agent_num = mutation_agent_num
        self.agent_count = len(init_agents)
        self.potential_score_EMA_coef = potential_score_EMA_coef

        self.children_num = defaultdict(int)

    def update_scores(self, rollouts):
        sampled_ids = []
        for i in range(cfg.PPO.NUM_ENVS):
            episode_end_index = torch.where(rollouts.masks[:, i] == 0)[0].cpu().numpy()
            start_t = 0
            for end_t in episode_end_index:
                gae = rollouts.ret[start_t:(end_t + 1), i] - rollouts.val[start_t:(end_t + 1), i]
                new_score = gae.abs().mean().item()
                agent_id = rollouts.unimal_ids[start_t, i].item()
                sampled_ids.append(agent_id)
                self.potential_score[agent_id] = self.potential_score[agent_id] * (1. - self.potential_score_EMA_coef) + new_score * self.potential_score_EMA_coef
                start_t = end_t + 1
            # add the final partial episode
            # don't do this if the partial episode is too short
            if cfg.PPO.TIMESTEPS - start_t >= 100:
                gae = rollouts.ret[start_t:, i] - rollouts.val[start_t:, i]
                new_score = gae.abs().mean().item()
                agent_id = rollouts.unimal_ids[start_t, i].item()
                self.potential_score[agent_id] = self.potential_score[agent_id] * (1. - self.potential_score_EMA_coef) + new_score * self.potential_score_EMA_coef
        # update staleness score
        # TODO: staleness score could be computed as EMA of the visited time before?
        sampled_ids = list(set(sampled_ids))
        self.staleness_score += 1.
        self.staleness_score[sampled_ids] = 0.

    def generate_new_agents(self, parent_agents, agent_scores):
        # TODO: how to effectively select parent agents
        # parent_agents = list(parent_agents)[:self.max_parent_num]
        if cfg.UED.PARENT_SELECT_STRATEGY == 'immediate_children':
            parents = self.agents
            # prefer agents with less children
            children_num = np.array([self.children_num[x] for x in parents])
            # prefer agents with smaller depth
            depth = [(len(x.split('-')) - 8) // 2 + 1 for x in parents]
            probs = 1. / ((children_num + 1.) * depth)
            probs /= probs.sum()
        else:
            parents = list(parent_agents)
            probs = np.ones(len(parent_agents)) / len(parent_agents)
        
        # do not mutate robots that have low scores
        if cfg.UED.MUTATE_THRESHOLD is not None:
            probs[agent_scores < cfg.UED.MUTATE_THRESHOLD] = 0.
            if probs.sum() == 0:
                return [], []
            probs /= probs.sum()

        print ('parent probs')
        print (probs)
        
        new_agents, new_agents_parent = [], []
        valid_mutation_num = 0
        while (1):
            # randomly sample a parent agent
            agent = parents[np.random.choice(len(parents), 1, p=probs)[0]]
            new_agent = f'{agent}-mutate-{self.agent_count}'
            unimal = SymmetricUnimal(new_agent, f'{cfg.ENV.WALKER_DIR}/unimal_init/{agent}.pkl')
            unimal.mutate()
            if unimal.num_limbs > 12 or len(xu.find_elem(unimal.actuator, "motor")) > 16:
                # discard current mutation if it is too large
                continue
            else:
                valid_mutation_num += 1
                self.agent_count += 1
                unimal.save()
                unimal.save_image()
                pickle_to_json(cfg.ENV.WALKER_DIR, unimal.id)
                new_agents.append(new_agent)
                new_agents_parent.append(agent)
                self.children_num[agent] += 1
            if valid_mutation_num == self.mutation_agent_num:
                break
        self.agents.extend(new_agents)
        return new_agents, new_agents_parent

    def initialize_new_agents_score(self, new_agents, potential_score=None):
        # get new agent's initial potential and staleness score
        # for now take an easiest approach of setting learning potential score the same as its parent's
        # TODO: think of a better way to do this
        if potential_score is None:
            new_potential_score = np.ones(len(new_agents)) * 100.
        else:
            new_potential_score = np.array(potential_score)
        self.potential_score = np.concatenate([self.potential_score, new_potential_score])
        new_staleness_score = np.zeros(len(new_agents))
        self.staleness_score = np.concatenate([self.staleness_score, new_staleness_score])

    def get_sample_probs(self, cur_iter):
        # normalize learning potential score by rank as in PLR paper
        # order = np.argsort(-self.potential_score)
        # rank = np.empty_like(order)
        # rank[order] = np.arange(len(order)) + 1
        # normalized_potential_score = 1. / rank
        # normalized_potential_score /= (normalized_potential_score).sum()
        normalized_potential_score = np.clip(self.potential_score, 0., 1.)
        normalized_potential_score /= (normalized_potential_score).sum()

        # compute stateless score
        # normalized_staleness_score = cur_iter - self.staleness_score + 1.
        normalized_staleness_score = self.staleness_score + 1.
        normalized_staleness_score /= normalized_staleness_score.sum()
        # combine them together
        probs = (1. - self.staleness_score_weight) * normalized_potential_score + self.staleness_score_weight * normalized_staleness_score
        return probs