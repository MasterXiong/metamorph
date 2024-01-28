import torch
import sys

from metamorph.config import cfg
from metamorph.algos.ppo.model import ActorCritic


def restore_from_checkpoint(ac, cp_path=None):
    if cp_path is None:
        cp_path = cfg.PPO.CHECKPOINT_PATH
    checkpoint = torch.load(cp_path)
    ob_rms, ret_rms, optimizer_state = None, None, None
    if len(checkpoint) == 2:
        model_p, ob_rms = checkpoint
    elif len(checkpoint) == 3:
        model_p, ob_rms, ret_rms = checkpoint
    else:
        model_p, ob_rms, ret_rms, optimizer_state = checkpoint

    if type(model_p) == ActorCritic:
        state_dict_p = model_p.state_dict()
    else:
        state_dict_p = model_p
    if any(['mu_net' in key for key in state_dict_p.keys()]):
        state_dict_c = ac.state_dict()
    else:
        state_dict_c = ac.mu_net.state_dict()
    if set(state_dict_c.keys()) != set(state_dict_p.keys()):
        print ('The params are not aligned!')
        print ('The following params are in model, but not in checkpoint:')
        print (set(state_dict_c.keys()) - set(state_dict_p.keys()))
        print ('The following params are in checkpoint, but not in model:')
        print (set(state_dict_p.keys()) - set(state_dict_c.keys()))
        sys.exit()

    fine_tune_layers = set()
    layer_substrings = cfg.MODEL.FINETUNE.LAYER_SUBSTRING
    # print (state_dict_p.keys())
    for name, param in state_dict_c.items():

        if any(name_substr in name for name_substr in layer_substrings):
            fine_tune_layers.add(name)
        
        # do not copy parameters for separate PE
        if 'separate_PE_encoder' in name:
            continue

        if name in state_dict_p:
            param_p = state_dict_p[name]
        else:
            print (f'the model does not have {name}')
            continue

        if param_p.shape == param.shape:
            with torch.no_grad():
                param.copy_(param_p)
        else:
            print (name, param_p.shape, param.shape)
            raise ValueError(
                "Checkpoint path is invalid as there is shape mismatch"
            )

    if not cfg.MODEL.FINETUNE.FULL_MODEL:
        for name, param in ac.named_parameters():
            if name not in fine_tune_layers:
                param.requires_grad = False
            else:
                print (f'fine tune {name}')
    else:
        print ('fine tune the whole model')

    return ob_rms, ret_rms, optimizer_state
