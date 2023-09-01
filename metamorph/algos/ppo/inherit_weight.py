import torch

from metamorph.config import cfg


def restore_from_checkpoint(ac):
    checkpoint = torch.load(cfg.PPO.CHECKPOINT_PATH)
    if len(checkpoint) == 2:
        model_p, ob_rms = checkpoint
        ret_rms = None
    else:
        model_p, ob_rms, ret_rms = checkpoint

    state_dict_c = ac.state_dict()
    state_dict_p = model_p.state_dict()

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

    if ret_rms:
        return ob_rms, ret_rms
    else:
        return ob_rms
