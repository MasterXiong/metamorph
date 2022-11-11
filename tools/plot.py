import json
import pickle
import torch
import matplotlib.pyplot as plt
import os


def run():
    # useful_agents = os.listdir('output/log_single_task_wo_pe')
    all_agents = os.listdir('output/log_single_task')
    for agent in all_agents:
        # if agent not in useful_agents:
            # os.system(f'rm -r output/log_single_task/{agent}')
        os.system(f'rm output/log_single_task/{agent}/*.pt')
        os.system(f'rm output/log_single_task/{agent}/*.yaml')
        os.system(f'rm output/log_single_task/{agent}/*.json')
        os.system(f'rm -r output/log_single_task/{agent}/tensorboard')


def compare_ST_training():
    agent_names = os.listdir('output/log_single_task_wo_pe')
    for agent in agent_names:
        print (agent)
        try:
            curve_baseline, curve_wo_pe = [], []
            for seed in ['1409', '1410', '1411']:
                if seed in os.listdir(f'output/log_single_task/{agent}'):
                    with open(f'output/log_single_task/{agent}/{seed}/Unimal-v0_results.json', 'r') as f:
                        result_baseline = json.load(f)
                    curve_baseline.append(result_baseline[agent]['reward']['reward'])
                if seed in os.listdir(f'output/log_single_task_wo_pe/{agent}'):
                    with open(f'output/log_single_task_wo_pe/{agent}/{seed}/Unimal-v0_results.json', 'r') as f:
                        result_wo_pe = json.load(f)
                    curve_wo_pe.append(result_wo_pe[agent]['reward']['reward'])
            
            plt.figure()
            for c in curve_baseline:
                plt.plot([i*2560*32 for i in range(len(c))], c, label='baseline', c='red')
            for c in curve_wo_pe:
                plt.plot([i*2560*32 for i in range(len(c))], c, label='wo pe', c='blue')
            plt.legend()
            plt.xlabel('Time step')
            plt.ylabel('Returns')
            plt.title(agent)
            plt.savefig(f'figures/ST_train_{agent}.png')
            plt.close()
        except:
            continue
        

def compare_train_curve():
    folders = ['log_origin', 
            #    'log_train_wo_task_id_pe', 
            #    'log_train_wo_task_id_pe_1410', 
            #    'log_train_wo_task_id_pe_1411', 
            #    'log_train_wo_task_id', 
            #    'log_train_wo_task_id_1410', 
               'log_train_wo_pe/1411', 
            #    'log_train_wo_task_id', 
            #    'log_train_wo_task_id_1410', 
            #    'log_hypernet_1410', 
            #    'log_hypernet', 
            # 'log_hypernet_pe_input_1410', 
            # 'log_hypernet_pe_input_1411', 
            #    'log_hypernet_hfi_1410', 
            #    'log_hypernet_hfi_1411', 
            #    'log_train_wo_task_id_pe_1411', 
            #    'log_hypernet', 
            #    'log_hypernet_new_init_1410', 
            #    'log_hypernet_new_init_1411', 
            #    'log_hypernet_no_init_1410', 
            #    'log_hypernet_no_init_1411', 
            #    'log_hypernet_1410_context_embed_32', 
            #    'log_hypernet+pe_1410', 
            #    'log_hypernet+pe_1411', 
            #    'log_baseline_wo_context/1409', 
            #    'log_baseline_wo_context/1410', 
            #    'log_baseline_wo_context_origin/1409', 
            # 'test_HN_with_context_normlization_wo_embed_scale_wo_es', 
            # 'test_HN_with_context_normalization_wo_embed_scale', 
            # 'test_HN_with_context_normalization_wo_embed_scale_kl_1.', 
            'test_HN_context_constant_norm_kl_1.', 
            'test_HN_context_constant_norm_wo_es',
            'test_HN_context_constant_norm', 
            # 'test_HN_PE_as_context_es_1', 
            # 'log_baseline_agrim_NM_feature/1410',  
            ]

    plt.figure()
    for folder in folders:
        with open(f'output/{folder}/Unimal-v0_results.json', 'r') as f:
            train_results = json.load(f)
        plt.plot(train_results['__env__']['reward']['reward'], label=folder)
        print (train_results['__env__']['reward']['reward'][-1])
    plt.legend()
    plt.savefig('figures/train_curve.png')
    plt.close()

    # path = 'output/log_hypernet_1410/checkpoint_500.pt'
    # m, _ = torch.load(path)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # w = m.v_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.subplot(1, 2, 2)
    # w = m.mu_net.hnet.weight.data
    # plt.hist(w.cpu().numpy().ravel())
    # plt.savefig('figures/hypernet_weight.png')
    # plt.close()



if __name__ == '__main__':
    # compare_ST_training()
    compare_train_curve()
    # run()