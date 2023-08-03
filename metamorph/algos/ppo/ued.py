import pickle
import json
import numpy as np
import copy
from collections import defaultdict

from metamorph.config import cfg

from derl.envs.morphology import SymmetricUnimal
from derl.utils import xml as xu


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


class ACCEL:

    def __init__(
        self, 
        init_agents, 
        staleness_score_weight=0.1, 
        mutation_agent_num=100, 
    ):
        # TODO: maybe need to use some optimistic intialization for potential score
        self.agents = copy.deepcopy(init_agents)
        self.potential_score = np.zeros(len(init_agents))
        self.staleness_score = np.zeros(len(init_agents))
        self.staleness_score_weight = staleness_score_weight
        self.mutation_agent_num = mutation_agent_num
        self.agent_count = len(init_agents)

        self.children_num = defaultdict(int)

    def generate_new_agents(self, parent_agents, agent_scores):
        # TODO: how to effectively select parent agents
        # parent_agents = list(parent_agents)[:self.max_parent_num]
        if cfg.UED.PARENT_SELECT_STRATEGY == 'immediate_children':
            parents = self.agents
            # prefer agents with less children
            children_num = np.array([self.children_num[x] for x in parents])
            # prefer agents closer to the root
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
        order = np.argsort(-self.potential_score)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order)) + 1
        normalized_potential_score = 1. / rank
        normalized_potential_score /= (normalized_potential_score).sum()
        # compute stateless score
        normalized_staleness_score = cur_iter - self.staleness_score + 1.
        normalized_staleness_score /= normalized_staleness_score.sum()
        # combine them together
        probs = (1. - self.staleness_score_weight) * normalized_potential_score + self.staleness_score_weight * normalized_staleness_score
        return probs