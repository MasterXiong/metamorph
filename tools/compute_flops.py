import matplotlib.pyplot as plt

from metamorph.config import cfg


def flops_linear(input_dim, output_dim):
    return 2 * input_dim * output_dim

def flops_mlp(limb_num, obs_dim, hidden_dim=256, hidden_layer=3, hfield=False):
    flops = 0
    # input layer
    flops += flops_linear(limb_num * obs_dim, hidden_dim)
    # hfield encoder
    if hfield:
        flops += flops_linear(1410, 64) + flops_linear(64, 64)
    # hidden layers
    dims = [hidden_dim for _ in range(hidden_layer)]
    if hfield:
        dims[0] += 64
    for i in range(hidden_layer - 1):
        flops += flops_linear(dims[i], dims[i + 1])
    # output layers
    flops += flops_linear(dims[-1], limb_num * 2)
    return flops

def flops_attention_layer(limb_num, embed_dim=128, nhead=2, linear_dim=1024, fix_attention=False):
    flops = 0
    # flops of single-head attention
    # flops for value computation
    attention_flops = limb_num * flops_linear(embed_dim, embed_dim)
    # flops for key, query and attention matrix
    if not fix_attention:
        attention_flops *= 3
        attention_flops += limb_num * limb_num * flops_linear(embed_dim, 1)
    # flops for aggregating values
    attention_flops += limb_num * limb_num * embed_dim * 2
    flops += attention_flops * nhead
    # flops for aggregation multi-head
    flops += limb_num * flops_linear(embed_dim * nhead, embed_dim)
    # flops for linear layers
    flops += limb_num * (flops_linear(embed_dim, linear_dim) + flops_linear(linear_dim, embed_dim))
    return flops

def flops_TF(obs_dim=52, layer_num=5, limb_num=12, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[64], fix_attention=False, hfield=False):
    flops = 0
    # embed
    flops += limb_num * flops_linear(obs_dim, embed_dim)
    # attention
    flops += layer_num * flops_attention_layer(limb_num, embed_dim, nhead, linear_dim, fix_attention)
    # hfield
    if hfield:
        flops += flops_linear(1410, 64) + flops_linear(64, 64)
    # decoder
    dims = [embed_dim] + decoder_dim + [2]
    if hfield:
        dims[0] += 64
    for i in range(len(dims) - 1):
        flops += limb_num * flops_linear(dims[i], dims[i + 1])
    return flops



limb_num = 12
print ('FT')
hnmlp_flops = flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=2)
print ('HN-MLP', hnmlp_flops)
tf_flops = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], fix_attention=False, hfield=False)
print ('TF', tf_flops, tf_flops / hnmlp_flops)
modumorph_flops = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], fix_attention=True, hfield=False)
print ('ModuMorph', modumorph_flops, modumorph_flops / hnmlp_flops)
modumorph_flops = flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=128, decoder_dim=[], fix_attention=True, hfield=False)
print ('Compressed ModuMorph', modumorph_flops, modumorph_flops / hnmlp_flops)

mlp_flops = []
tf_flops = []
for limb_num in range(5, 16):
    mlp_flops.append(flops_mlp(limb_num=limb_num, obs_dim=17, hidden_dim=256, hidden_layer=2))
    tf_flops.append(flops_TF(obs_dim=52, layer_num=5, limb_num=limb_num, embed_dim=128, nhead=2, linear_dim=1024, decoder_dim=[], fix_attention=False, hfield=False))
plt.figure()
plt.plot(list(range(5, 16)), mlp_flops)
plt.plot(list(range(5, 16)), tf_flops)
plt.savefig('figures/efficiency.png')
plt.close()