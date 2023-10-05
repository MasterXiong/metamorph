import numpy as np
import os
import pickle
import json
from collections import defaultdict
from multiprocessing import Pool

from derl.config import dump_cfg
from derl.envs.morphology import SymmetricUnimal
from derl.utils import evo as eu
from derl.utils import file as fu
from derl.utils import sample as su
from derl.utils import similarity as simu

from lxml import etree

from derl.utils import geom as gu
from derl.utils import mjpy as mu
from derl.utils import xml as xu

from metamorph.config import cfg


def setup_output_dir():
    print (cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Make subfolders
    subfolders = [
        "models",
        "metadata",
        "xml",
        "unimal_init",
        "rewards",
        "videos",
        "error_metadata",
        "images",
    ]
    for folder in subfolders:
        os.makedirs(os.path.join(cfg.OUT_DIR, folder), exist_ok=True)


def sample_morphology(unimal_id, folder):
    # Build unimals which initialize the population based on number of limbs.
    unimal = SymmetricUnimal(unimal_id)
    # num_limbs = su.sample_from_range(cfg.LIMB.NUM_LIMBS_RANGE)
    np.random.seed(int(unimal_id))
    num_limbs = np.random.choice(np.arange(5, 11), 1)[0]
    print ('max limb num', num_limbs)
    unimal.mutate(op="grow_limb")
    while unimal.num_limbs < num_limbs:
        unimal.mutate()

    unimal.save(folder)
    pickle_to_json(folder, unimal_id)
    # unimal.save_image()
    return unimal_id


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


def xml_to_pickle(folder, name):

    with open(f'{folder}/metadata/{name}.json', 'r') as f:
        metadata = json.load(f)

    root, tree = xu.etree_from_xml(f'{folder}/xml/{name}.xml')
    worldbody = root.findall("./worldbody")[0]
    actuator = root.findall("./actuator")[0]
    contact = root.findall("./contact")[0]
    
    unimal = xu.find_elem(root, "body", attr_type="name", attr_value='torso/0')[0]
    sim = mu.mjsim_from_etree(root)

    limbs = xu.find_elem(unimal, "body")
    limb_names = [limb.get('name') for limb in limbs]
    limb_ids = [int(x.split('/')[1]) for x in limb_names]

    states = {}
    states['xml_path'] = f'{folder}/xml/{name}.xml'
    states['num_limbs'] = len(limbs)
    states['limb_idx'] = max(limb_ids) + 1

    # find symmetric limbs with the metadata json file
    limb_list = metadata['symmetric_limbs'].copy()
    symmetric_limbs = []
    for x in limb_list:
        symmetric_limbs.extend(x)
    for limb in limb_ids:
        if limb not in symmetric_limbs:
            limb_list.append([limb])
    states['limb_list'] = limb_list

    # add parent and child to contact pairs
    contact_pairs = []
    for limb in limbs:
        parent = limb.getparent()
        contact_pairs.append((parent.get('name'), limb.get('name'), ))
    # add children with the same parent slot to contact pairs
    for limb_pair in metadata['symmetric_limbs']:
        i = limb_ids.index(limb_pair[0])
        j = limb_ids.index(limb_pair[1])
        if limbs[i].getparent() == limbs[j].getparent():
            contact_pairs.append((limb_names[i], limb_names[j], ))
    states['contact_pairs'] = set(contact_pairs)

    # add limb metadata
    limb_metadata = defaultdict(dict)
    for limb, idx in zip(limbs, limb_ids):
        # placeholder to store joint information
        limb_metadata[idx]['joint_axis'] = ''
        limb_metadata[idx]['gear'] = {}
        # orient
        # Issue: theta can not be recovered if sin(phi)=0
        geom_property = xu.find_elem(limb, 'geom', child_only=True)[0]
        coordinate = xu.str2arr(geom_property.get('fromto'))
        x, y, z = coordinate[-3:]
        h_plus_r, theta, phi = gu.cart2sph(x, y, z)
        h = h_plus_r - float(geom_property.get('size'))
        h = np.array([0.2, 0.3, 0.4])[np.argmin(np.abs(h - np.array([0.2, 0.3, 0.4])))]
        limb_metadata[idx]['orient'] = (h, theta, phi)
        # parent_name
        parent = limb.getparent()
        limb_metadata[idx]['parent_name'] = parent.get('name')
        # site
        candidate_sites = xu.find_elem(parent, 'site', attr_type='class', attr_value='growth_site', child_only=True)
        mirror_candidate_sites = xu.find_elem(parent, 'site', attr_type='class', attr_value='mirror_growth_site', child_only=True)
        candidate_sites.extend(mirror_candidate_sites)
        limb_pos = xu.str2arr(limb.get('pos'))
        parent_radius = float(xu.find_elem(parent, 'geom', child_only=True)[0].get("size"))
        distance = []
        for site in candidate_sites:
            site_pos = xu.str2arr(site.get("pos"))
            attach_pos = xu.add_list(site_pos, gu.sph2cart(parent_radius, theta, phi))
            distance.append(((np.array(attach_pos) - np.array(limb_pos)) ** 2).sum())
        site_idx = np.argmin(np.array(distance))
        limb_metadata[idx]['site'] = candidate_sites[site_idx].get('name')
 
    # add joints metadata
    for joint in actuator.getchildren():
        limb_id = int(joint.get('name').split('/')[-1])
        axis = joint.get('name').split('/')[0][-1]
        limb_metadata[limb_id]['joint_axis'] += axis
        limb_metadata[limb_id]['gear'][axis] = int(joint.get('gear'))
    states['limb_metadata'] = limb_metadata

    # mirror_sites
    states['mirror_sites'] = {}
    for limb_pair in limb_list:
        if len(limb_pair) == 1:
            continue
        limb_0, limb_1 = limbs[limb_ids.index(limb_pair[0])], limbs[limb_ids.index(limb_pair[1])]
        sites_0 = xu.find_elem(limb_0, 'site', attr_type='class', attr_value='mirror_growth_site', child_only=True)
        sites_1 = xu.find_elem(limb_1, 'site', attr_type='class', attr_value='mirror_growth_site', child_only=True)
        for site_0, site_1 in zip(sites_0, sites_1):
            states['mirror_sites'][site_0.get('name')] = site_1.get('name')
            states['mirror_sites'][site_1.get('name')] = site_0.get('name')

    # body_params
    states["body_params"] = {}
    states["body_params"]['torso_mode'] = xu.find_elem(unimal, 'site', attr_type='class', attr_value='torso_growth_site', child_only=True)[0].get('name').split('/')[1]
    states["body_params"]['torso_density'] = float(xu.find_elem(unimal, 'geom', child_only=True)[0].get('density'))
    states["body_params"]['limb_density'] = float(xu.find_elem(limbs[0], 'geom', child_only=True)[0].get('density'))
    states["body_params"]['num_torso'] = 0

    # others
    states["num_torso"] = 1
    states["torso_list"] = [0]
    states["growth_torso"] = [0]
    states["dof"] = metadata["dof"]

    with open(f'{folder}/unimal_init/{name}.pkl', 'wb') as f:
        pickle.dump(states, f)

    return states


def find_removable_limbs(agent):
    limb_list = agent.limb_list.copy()
    removable_limbs = []
    for limbs in limb_list:
        body = xu.find_elem(agent.unimal, "body", "name", "limb/{}".format(limbs[0]))[0]
        num_children = len(xu.find_elem(body, "body", child_only=True))
        if num_children == 0:
            removable_limbs.append(limbs)
    return removable_limbs


def generate_one_level_limb_remove(source_folder, target_folder):
    # generate all possible mutations by removing one limb, or a pair of symmetric limbs
    cfg.ENV.WALKER_DIR = target_folder
    os.system(f'rm -r {target_folder}')
    os.system(f'cp -r {source_folder} {target_folder}')
    for xml in os.listdir(f'{source_folder}/xml'):
        agent = xml[:-4]
        print (agent)
        unimal = SymmetricUnimal(agent, f'{target_folder}/unimal_init/{agent}.pkl')
        if unimal.num_limbs > 2:
            limbs = find_removable_limbs(unimal)
            for i, l in enumerate(limbs):
                name = f'{agent}-mutate-{i}'
                unimal = SymmetricUnimal(name, f'{target_folder}/unimal_init/{agent}.pkl')
                unimal.mutate_delete_limb(limb_to_remove=l)
                unimal.save()
                unimal.save_image()
                pickle_to_json(target_folder, unimal.id)
        os.system(f'rm {target_folder}/xml/{xml}')
        os.system(f'rm {target_folder}/images/{agent}.png')
        os.system(f'rm {target_folder}/metadata/{agent}.json')
        os.system(f'rm {target_folder}/unimal_init/{agent}.pkl')



if __name__ == "__main__":

    # sample a random set of robots
    folder = 'random_100k'
    os.makedirs(f'data/{folder}', exist_ok=True)
    os.makedirs(f'data/{folder}/xml', exist_ok=True)
    os.makedirs(f'data/{folder}/unimal_init', exist_ok=True)
    os.makedirs(f'data/{folder}/metadata', exist_ok=True)
    with Pool(processes=20) as pool:
        pool.starmap(sample_morphology, [(str(i), f'data/{folder}') for i in range(200000)])
    # for i in range(100000):
    #     sample_morphology(str(i), f'data/{folder}')

    # folder = 'unimals_100/train_valid_1409'
    # for agent in os.listdir(f'{folder}/xml'):
    #     root, tree = xu.etree_from_xml(f'{folder}/xml/{agent}')
    #     # worldbody = root.findall("./worldbody")[0]
    #     actuator = root.findall("./actuator")[0]
    #     # contact = root.findall("./contact")[0]
    #     dofs = xu.find_elem(actuator, "motor")
    #     if len(dofs) == 0:
    #         print (agent)
    # for agent in os.listdir(f'{folder}/metadata'):
    #     with open(f'{folder}/metadata/{agent}', 'r') as f:
    #         data = json.load(f)
    #     if data['num_limbs'] == 10:
    #         print (agent)
    #         print (data)
    # os.system('rm unimals_100/train/metadata/*')
    # os.system('rm unimals_100/train/unimal_init/*')
    # os.system('rm unimals_100/train/xml/*')
    # for agent in os.listdir('unimals_100/train/images'):
    #     name = agent[:-4]
    #     os.system(f'cp unimals_100/train_mutate_400/metadata/{name}.json unimals_100/train/metadata/')
    #     os.system(f'cp unimals_100/train_mutate_400/unimal_init/{name}.pkl unimals_100/train/unimal_init/')
    #     os.system(f'cp unimals_100/train_mutate_400/xml/{name}.xml unimals_100/train/xml/')

    # generate_one_level_limb_remove('unimals_100/train_remove_level_1', 'unimals_100/train_remove_level_2')
    # generate_one_level_limb_remove('unimals_100/train_remove_level_2', 'unimals_100/train_remove_level_3')

    # cfg.ENV.WALKER_DIR = 'morphology'
    # cfg.OUT_DIR = 'morphology'
    # setup_output_dir()

    # # check xml-pickle converter
    # for i in range(10):
    #     sample_morphology(i)
    #     pickle_to_json(cfg.ENV.WALKER_DIR, i)
    #     with open(f'morphology/unimal_init/{i}.pkl', 'rb') as f:
    #         states_true = pickle.load(f)
    #     states = xml_to_pickle('morphology', i)
    #     # for key in states_true:
    #     #     print (key)
    #     #     print ('ground truth')
    #     #     print (states_true[key])
    #     #     print ('reconstructed')
    #     #     print (states[key])
    #     for limb in states_true['limb_metadata']:
    #         print (limb)
    #         for subkey in ['joint_axis', 'orient', 'parent_name', 'site', 'gear']:
    #             print (subkey, states_true['limb_metadata'][limb][subkey], states['limb_metadata'][limb][subkey])

    # folder = 'data/expand_base'
    # os.makedirs(f'{folder}/unimal_init', exist_ok=True)
    # # os.makedirs(f'{folder}/images', exist_ok=True)
    # for xml in os.listdir(f'{folder}/xml'):
    #     agent = xml[:-4]
    #     xml_to_pickle(folder, agent)
        # unimal = SymmetricUnimal(agent, f'{folder}/unimal_init/{agent}.pkl')
        # unimal.save_image(folder)
    
    # mutate existing training robots
    # source_folder = 'unimals_100/train'
    # target_folder = 'unimals_100/train_mutate_100_v2'
    # cfg.OUT_DIR = target_folder
    # # os.system(f'rm -r {target_folder}')
    # os.system(f'cp -r {source_folder} {target_folder}')
    # max_mutation_num = 3
    # mutation_per_robot = 1
    # for xml in os.listdir(f'{source_folder}/xml'):
    #     agent = xml[:-4]
    #     valid_count = 0
    #     while valid_count < mutation_per_robot:
    #         mutation_num = np.random.choice(max_mutation_num, 1)[0] + 1
    #         name = f'{agent}-mutate-{valid_count}'
    #         unimal = SymmetricUnimal(name, f'{target_folder}/unimal_init/{agent}.pkl')
    #         for _ in range(mutation_num):
    #             unimal.mutate()
    #             unimal.id += f'-{unimal.curr_mutation}'
    #         if unimal.num_limbs > 12 or len(xu.find_elem(unimal.actuator, "motor")) > 16:
    #             # discard to current mutation if it is too large
    #             continue
    #         else:
    #             valid_count += 1
    #             unimal.save(target_folder)
    #             # unimal.save_image()
    #             pickle_to_json(target_folder, unimal.id)
    #     os.system(f'rm {target_folder}/xml/{xml}')
    #     os.system(f'rm {target_folder}/metadata/{agent}.json')
    #     os.system(f'rm {target_folder}/unimal_init/{agent}.pkl')
    #     os.system(f'rm {target_folder}/images/{agent}.jpg')

    # mutate existing training robots
    # source_folder = 'unimals_100/train'
    # for op in ['grow_limb', 'delete_limb']:
    #     target_folder = f'unimals_100/train_mutate_{op}'
    #     cfg.ENV.WALKER_DIR = target_folder
    #     os.system(f'rm -r {target_folder}')
    #     os.system(f'cp -r {source_folder} {target_folder}')
    #     mutation_per_robot = 3
    #     for xml in os.listdir(f'{source_folder}/xml'):
    #         agent = xml[:-4]
    #         for i in range(mutation_per_robot):
    #             name = f'{agent}-mutate-{i}'
    #             unimal = SymmetricUnimal(name, f'{target_folder}/unimal_init/{agent}.pkl')
    #             unimal.mutate(op=op)
    #             unimal.save()
    #             unimal.save_image()
    #             pickle_to_json(target_folder, unimal.id)
    
    # for xml in os.listdir(f'unimals_100/train_remove_level_1/xml'):
    #     agent = xml[:-4]
    #     for subfolder, suffix in zip(['images', 'metadata', 'unimal_init', 'xml'], ['.jpg', '.json', '.pkl', '.xml']):
    #         os.system(f'rm unimals_100/train_remove_level_1/{subfolder}/{agent}{suffix}')
 
    # generate robots with only 3 limbs
    # source_folder = 'unimals_100/expand_base'
    # target_folder = 'unimals_100/expand_level_1'
    # os.makedirs(f'{target_folder}', exist_ok=True)
    # os.makedirs(f'{target_folder}/images', exist_ok=True)
    # os.makedirs(f'{target_folder}/metadata', exist_ok=True)
    # os.makedirs(f'{target_folder}/unimal_init', exist_ok=True)
    # os.makedirs(f'{target_folder}/xml', exist_ok=True)
    # # for unimal_id in range(100):
    # #     unimal = SymmetricUnimal(unimal_id)
    # #     while unimal.num_limbs not in [3, 4]:
    # #         unimal.mutate(op="grow_limb")
    # #     unimal.save(folder)
    # #     unimal.save_image(folder)
    # parent_num = len(os.listdir(f'{source_folder}/xml'))
    # for i in range(parent_num):
    #     # pickle_to_json(folder, str(i))
    #     for j in range(3):
    #         idx = i * 3 + j + parent_num
    #         unimal = SymmetricUnimal(str(idx), f'{source_folder}/unimal_init/{i}.pkl')
    #         unimal.mutate(op="grow_limb")
    #         unimal.save(target_folder)
    #         unimal.save_image(target_folder)
    #         pickle_to_json(target_folder, unimal.id)

    # source_folder = 'unimals_100/random_1000'
    # target_folder = 'unimals_100/random_100_v2'
    # os.makedirs(f'{target_folder}', exist_ok=True)
    # os.makedirs(f'{target_folder}/images', exist_ok=True)
    # os.makedirs(f'{target_folder}/metadata', exist_ok=True)
    # os.makedirs(f'{target_folder}/unimal_init', exist_ok=True)
    # os.makedirs(f'{target_folder}/xml', exist_ok=True)

    # count = 0
    # for xml in os.listdir(f'{source_folder}/xml'):
    #     if xml not in os.listdir('unimals_100/random_100/xml'):
    #         agent = xml[:-4]
    #         os.system(f'cp {source_folder}/xml/{xml} {target_folder}/xml/')
    #         os.system(f'cp {source_folder}/metadata/{agent}.json {target_folder}/metadata/')
    #         count += 1
    #     if count == 100:
    #         break

    # source_folder = 'unimals_100/random_100'
    # target_folder = 'unimals_100/random_100_mutate_1'
    # os.makedirs(f'{target_folder}', exist_ok=True)
    # os.makedirs(f'{target_folder}/images', exist_ok=True)
    # os.makedirs(f'{target_folder}/metadata', exist_ok=True)
    # os.makedirs(f'{target_folder}/unimal_init', exist_ok=True)
    # os.makedirs(f'{target_folder}/xml', exist_ok=True)
    # for unimal_id in range(100):
    #     unimal = SymmetricUnimal(unimal_id)
    #     while unimal.num_limbs not in [3, 4]:
    #         unimal.mutate(op="grow_limb")
    #     unimal.save(folder)
    #     unimal.save_image(folder)
    # parent_num = len(os.listdir(f'{source_folder}/xml'))
    # for xml in os.listdir(f'{source_folder}/xml'):
    #     i = int(xml[:-4])
    #     idx = parent_num + i
    #     unimal = SymmetricUnimal(f'{idx}-parent-{i}', f'{source_folder}/unimal_init/{i}.pkl')
    #     unimal.mutate()
    #     unimal.save(target_folder)
    #     # unimal.save_image(target_folder)
    #     pickle_to_json(target_folder, unimal.id)

    # for xml in os.listdir('unimals_100/random_100_v2/xml'):
    #     agent = xml[:-4]
    #     xml_to_pickle('unimals_100/random_100_v2', agent)