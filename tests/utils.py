import torch

def load_dense(state_dict, sub_key, param):
    if sub_key == 'weight':
        weights = torch.from_numpy(state_dict['kernel']).T
    elif sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    param.copy_(weights)


def load_group_norm(state_dict, sub_key, param):
    if sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    elif sub_key == 'weight':
        weights = torch.from_numpy(state_dict['scale'])
    else:
        raise NotImplementedError(sub_key)
    param.copy_(weights.reshape(param.shape))


def load_conv(state_dict, sub_key, param):
    if sub_key == 'weight':
        weights = torch.from_numpy(state_dict['kernel'])
        # kernel: (kernel_size, kernel_size, C_in, C_out)
        # target: (C_out, C_in, kernel_size, kernel_size)
        weights = weights.permute(3, 2, 0, 1)
    elif sub_key == 'bias':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    num_ch = param.shape[0]
    param.copy_(weights[:num_ch])


def load_nin(state_dict, sub_key, param):
    if sub_key == 'W':
        weights = torch.from_numpy(state_dict['kernel']).squeeze()
    elif sub_key == 'b':
        weights = torch.from_numpy(state_dict['bias'])
    else:
        raise NotImplementedError
    param.copy_(weights.reshape(param.shape))



def load_res_block(state_dict,
                  block_key, sub_key, 
                  param):
    res_dict = {
        'GroupNorm_0': 'norm1',
        'Conv_0': 'conv1',
        'Dense_0': 'temb_proj', 
        'GroupNorm_1': 'norm2', 
        'Conv_1': 'conv2', 
        'NIN_0': 'nin_shortcut'
    }
    module_key = res_dict[block_key]
    module_dict = state_dict[module_key]

    if module_key in ['conv1', 'conv2']:
        load_conv(state_dict=module_dict, 
                  sub_key=sub_key, 
                  param=param)
    elif module_key in ['norm1', 'norm2']:
        load_group_norm(state_dict=module_dict, 
                        sub_key=sub_key, 
                        param=param)
    elif module_key == 'temb_proj':
        load_dense(state_dict=module_dict, 
                   sub_key=sub_key, 
                   param=param)
    else:
        load_nin(state_dict=module_dict, 
                 sub_key=sub_key, 
                 param=param)

    
def load_attn_block(state_dict, 
                    block_key, sub_key, 
                    param):
    attn_dict = {
        'NINs': 'qkv', 
        'NIN_3': 'proj_out', 
        'GroupNorm_0': 'norm'
    }

    qkv_sub_map = {
        'W': 'kernel', 
        'b': 'bias'
    }
    module_key = attn_dict[block_key]

    if module_key == 'norm':
        load_group_norm(state_dict=state_dict[module_key],
                        sub_key=sub_key, 
                        param=param)
    elif module_key == 'proj_out':
        load_nin(state_dict=state_dict[module_key], 
                 sub_key=sub_key, 
                 param=param)
    else:
        # load qkv
        qkv_sub_key = qkv_sub_map[sub_key]

        weight_list = []
        sub_shape = list(param.shape)
        sub_shape[-1] = sub_shape[-1] // 3
        for qkv_key in module_key:
            weight = torch.from_numpy(state_dict[qkv_key][qkv_sub_key]).reshape(sub_shape)
            weight_list.append(weight)
        qkv_weight = torch.cat(weight_list, dim=-1)
        param.copy_(qkv_weight)
