import numpy as onp
import jax.numpy as jnp
import jax
import torch

from flax import serialization

from flax import linen as nn

from models.ddpmm import NIN
from tests.utils import load_conv, load_nin, load_dense

_key_map = {
    'weight': 'kernel', 
    'bias': 'bias'
}


def run_jax_nn(params, module, in_x):
    x = jnp.asarray(in_x)
    out = module.apply(
        {'params': params}, x
    )
    out = jax.device_get(out)
    return out


def run_torch_nn(module, in_x):
    x = torch.from_numpy(in_x).permute(0, 3, 1, 2)
    out = module(x).permute(0, 2, 3, 1)
    out = out.numpy()
    return out


@torch.no_grad()
def test_conv(state_dict):
    onp.random.seed(0)
    key_name = 'conv_in'
    B, H, W, C = 2, 64, 64, 3
    # create modules
    out_ch = 192
    jax_nn = nn.Conv(features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv_in')

    torch_nn = torch.nn.Conv2d(C, out_ch, kernel_size=3, stride=1, padding=1)

    for name, param in torch_nn.named_parameters():
        load_conv(state_dict, sub_key=name, param=param)
        
    # create input
    x = onp.random.normal(size=(B, H, W, C)).astype(onp.float32)
    
    # results from jax module 
    jax_out = run_jax_nn(state_dict, jax_nn, x)

    torch_out = run_torch_nn(torch_nn, x)

    # check 
    onp.testing.assert_almost_equal(jax_out, torch_out, decimal=5)


@torch.no_grad()
def test_dense_general(state_dict):
    B, H, W, num_heads, head_dim = 2, 8, 8, 6, 64
    C = num_heads * head_dim
    onp.random.seed(0)
    key_name = 'down_1.attn_0'
    sub_key = 'proj_out'

    jax_nn = nn.DenseGeneral(
        features=C, 
        axis=(-2, -1), 
        name='proj_out'
    )

    torch_nn = NIN(C, C)
    for name, param in torch_nn.named_parameters():
        load_nin(state_dict, name, param)

    # create input
    x = onp.random.normal(size=(B, H, W, num_heads, head_dim)).astype(onp.float32)

    jx = x.reshape((B, H * W, num_heads, head_dim))
    jax_out = run_jax_nn(state_dict, jax_nn, jx)    # B, H * W, C

    # pytorch function

    init_x = torch.from_numpy(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)   # B, C, H, W
    out = torch_nn(init_x).permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C
    torch_out = out.reshape(B, H * W, C).numpy()

    onp.testing.assert_almost_equal(jax_out, torch_out, decimal=5)


@torch.no_grad()
def test_dense(state_dict):
    B, in_ch, emb_ch = 2, 768, 768
    key_name = 'dense1'

    onp.random.seed(0)

    jax_nn = nn.Dense(emb_ch, name='dense1')
    torch_nn = torch.nn.Linear(in_ch, emb_ch)

    for name, param in torch_nn.named_parameters():
        load_dense(state_dict, name, param)
    
    # set up input 
    x = onp.random.normal(size=(B, in_ch)).astype(onp.float32)

    jax_out = run_jax_nn(state_dict, jax_nn, x) # B, emb_ch

    torch_x = torch.from_numpy(x)
    out = torch_nn(torch_x)
    torch_out = out.numpy()
    onp.testing.assert_almost_equal(jax_out, torch_out, decimal=6)


if __name__ == '__main__':
    ckpt_path = 'ckpts/imagenet_original'
    with open(ckpt_path, 'rb') as f:
        ckpt = serialization.from_bytes(target=None, encoded_bytes=f.read())['ema_params']
    # key_name = 'conv_in'
    # test_conv(ckpt[key_name])

    # test dense general
    key_name = 'down_1.attn_0'
    sub_key = 'proj_out'
    test_dense_general(state_dict=ckpt[key_name][sub_key])

    # key_name = 'dense1'
    # test_dense(ckpt[key_name])