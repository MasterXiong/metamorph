import matplotlib.pyplot as plt

from metamorph.config import cfg


def flops_linear(input_dim, output_dim):
    return 2 * input_dim * output_dim

def params_linear(input_dim, output_dim):
    return (input_dim + 1) * output_dim

def flops_mlp(limb_num, obs_dim, hidden_dim=256, hidden_layer=3, hfield=False):
    flops, params = 0, 0
    # input layer
    flops += flops_linear(limb_num * obs_dim, hidden_dim)
    params = params_linear(limb_num * obs_dim, hidden_dim)
    # hfield encoder
    if hfield:
        flops += flops_linear(1410, 64) + flops_linear(64, 64)
        params += params_linear(1410, 64) + params_linear(64, 64)
    # hidden layers
    dims = [hidden_dim for _ in range(hidden_layer)]
    if hfield:
        dims[0] += 64
    for i in range(hidden_layer - 1):
        flops += flops_linear(dims[i], dims[i + 1])
        params += params_linear(dims[i], dims[i + 1])
    # output layers
    flops += flops_linear(dims[-1], limb_num * 2)
    params += params_linear(dims[-1], limb_num * 2)
    return flops, params

def flops_attention_layer(limb_num, embed_dim=128, nhead=2, linear_dim=1024, fix_attention=False):
    flops, params = 0, 0
    # flops of single-head attention
    # flops for value computation
    attention_flops = limb_num * flops_linear(embed_dim, embed_dim)
    attention_params = params_linear(embed_dim, embed_dim)
    # flops for key, query and attention matrix
    if not fix_attention:
        attention_flops *= 3
        attention_flops += limb_num * limb_num * flops_linear(embed_dim, 1)
        attention_params *= 3
    else:
        # params for the attention matrix
        attention_params += limb_num ** 2
    # flops for aggregating values
    attention_flops += limb_num * limb_num * embed_dim * 2
    # cost for nhead
    flops += attention_flops * nhead
    params += attention_params * nhead
    # flops for aggregation multi-head
    flops += limb_num * flops_linear(embed_dim * nhead, embed_dim)
    params += params_linear(embed_dim * nhead, embed_dim)
    # flops for linear layers
    flops += limb_num * (flops_linear(embed_dim, linear_dim) + flops_linear(linear_dim, embed_dim))
    params += params_linear(embed_dim, linear_dim) + params_linear(linear_dim, embed_dim)
    return flops, params

def flops_TF(obs_dim=52, layer_num=5, limb_num=12, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[64], modumorph=False, hfield=False):
    flops, params = 0, 0
    # embed
    flops += limb_num * flops_linear(obs_dim, embed_dim)
    if modumorph:
        params += limb_num * params_linear(obs_dim, embed_dim)
    else:
        params += params_linear(obs_dim, embed_dim)
    # attention
    attention_flops, attention_params = flops_attention_layer(limb_num, embed_dim, nhead, linear_dim, fix_attention=modumorph)
    flops += layer_num * attention_flops
    params += layer_num * attention_params
    # hfield
    if hfield:
        flops += flops_linear(1410, 64) + flops_linear(64, 64)
        params += params_linear(1410, 64) + params_linear(64, 64)
    # decoder
    dims = [embed_dim] + decoder_dim + [2]
    if hfield:
        dims[0] += 64
    for i in range(len(dims) - 1):
        flops += limb_num * flops_linear(dims[i], dims[i + 1])
        if modumorph:
            params += limb_num * params_linear(dims[i], dims[i + 1])
        else:
            params += params_linear(dims[i], dims[i + 1])
    return flops, params



limb_num = 12
print ('FT')
hnmlp_flops, hnmlp_params = flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=2)
print ('HN-MLP', hnmlp_flops / 1e6, hnmlp_params / 1e6)
mlp_flops, mlp_params = flops_mlp(limb_num=limb_num, obs_dim=52, hidden_dim=256, hidden_layer=2)
print ('MLP', mlp_flops / 1e6, mlp_flops / hnmlp_flops, mlp_params / 1e6, mlp_params / hnmlp_params)
# tf_flops = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], fix_attention=False, hfield=False)
# print ('TF', tf_flops, tf_flops / hnmlp_flops)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], modumorph=True, hfield=False)
print ('ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=1, limb_num=limb_num, embed_dim=128, nhead=1, linear_dim=128, decoder_dim=[], modumorph=True, hfield=False)
print ('Compressed ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)

print ('csr')
hnmlp_flops, hnmlp_params = flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=3, hfield=True)
print ('HN-MLP', hnmlp_flops / 1e6, hnmlp_params / 1e6)
mlp_flops, mlp_params = flops_mlp(limb_num=limb_num, obs_dim=52, hidden_dim=256, hidden_layer=3, hfield=True)
print ('MLP', mlp_flops / 1e6, mlp_flops / hnmlp_flops, mlp_params / 1e6, mlp_params / hnmlp_params)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[64], modumorph=True, hfield=True)
print ('ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=2, limb_num=limb_num, embed_dim=128, nhead=1, linear_dim=128, decoder_dim=[], modumorph=True, hfield=True)
print ('Compressed ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)

print ('obstacle')
hnmlp_flops, hnmlp_params = flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=3, hfield=True)
print ('HN-MLP', hnmlp_flops / 1e6, hnmlp_params / 1e6)
mlp_flops, mlp_params = flops_mlp(limb_num=limb_num, obs_dim=52, hidden_dim=256, hidden_layer=3, hfield=True)
print ('MLP', mlp_flops / 1e6, mlp_flops / hnmlp_flops, mlp_params / 1e6, mlp_params / hnmlp_params)
# tf_flops = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[64, 64], fix_attention=False, hfield=True)
# print ('TF', tf_flops, tf_flops / hnmlp_flops)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[64, 64], modumorph=True, hfield=True)
print ('ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)
modumorph_flops, modumorph_params = flops_TF(obs_dim=52, layer_num=2, limb_num=limb_num, embed_dim=128, nhead=1, linear_dim=128, decoder_dim=[], modumorph=True, hfield=True)
print ('ModuMorph', modumorph_flops / 1e6, modumorph_flops / hnmlp_flops, modumorph_params / 1e6, modumorph_params / hnmlp_params)

# mlp_flops = []
# tf_flops = []
# for limb_num in range(5, 16):
#     mlp_flops.append(flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=2))
#     tf_flops.append(flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], fix_attention=False, hfield=False))
# plt.figure()
# plt.plot(list(range(5, 16)), mlp_flops)
# plt.plot(list(range(5, 16)), tf_flops)
# plt.savefig('figures/efficiency.png')
# plt.close()