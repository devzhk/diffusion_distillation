import os
import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as onp
import jax
import jax.numpy as jnp
# import tensorflow as tf
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='False'
from flax import serialization
import flax
from PIL import Image
import diffusion_distillation
from diffusion_distillation import utils
import lmdb


def save2dir(images, outdir, curr):
    num_imgs = images.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(images[j])
        img_path = os.path.join(outdir, f'{j + curr}.jpg')
        im.save(img_path, quality=100, subsampling=0)    
    return curr + num_imgs


def save2db(traj_batch, env, curr):
    '''
    Input: 
        traj: ndarray, (B, T, C, H, W)
    '''
    num_traj = traj_batch.shape[0]
    with env.begin(write=True) as txn:
        for i in range(num_traj):
            key = f'{curr+i}'.encode()
            txn.put(key, traj_batch[i])
    return curr + num_traj


def generate_samples(args):
    batch = args.batchsize
    num_batchs = args.num_imgs // batch
    # sample from the model
    # create imagenet model
    config = diffusion_distillation.config.imagenet64_distill.get_config()
    model = diffusion_distillation.model.Model(config)

    with open(args.ckpt_path, 'rb') as f:
        loaded_params = serialization.from_bytes(target=None, encoded_bytes=f.read())['ema_params']
    # fix possible flax version errors
    ema_params = jax.device_get(model.make_init_state()).ema_params
    loaded_params = flax.core.unfreeze(loaded_params)
    loaded_params = jax.tree_map(
        lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
    loaded_params,
    flax.core.unfreeze(ema_params))
    loaded_params = flax.core.freeze(loaded_params)
    del ema_params
    imagenet_classes = {'malamute': 249, 'siamese': 284, 'great_white': 2,
                        'speedboat': 814, 'reef': 973, 'sports_car': 817,
                        'race_car': 751, 'model_t': 661, 'truck': 867}
    # labels = imagenet_classes['truck'] * jnp.ones((batch,), dtype=jnp.int32)
    outdir = 'exp/imagenet16/figs'
    os.makedirs(outdir, exist_ok=True)

    curr = 0
    for i in tqdm(range(num_batchs)):
        labels = jax.random.randint(jax.random.PRNGKey(i), (batch, ), minval=0, maxval=1000, dtype=jnp.int32)
        samples = model.samples_fn(rng=jax.random.PRNGKey(i), labels=labels, params=loaded_params, num_steps=16)
        samples = jax.device_get(samples).astype(onp.uint8)
        curr = save2dir(samples, outdir, curr)
    # outdir = 'data/imagenet8/lmdb'
    # samples = samples[:, None, :, :, :].repeat(9, axis=1)
    # env = lmdb.open(outdir, map_size=10*1024*1024*1024, readahead=False)
    # curr = save2db(samples, env, curr)
    
    print(curr)


if __name__ == '__main__':
    parser = ArgumentParser(description='parser for DDIM sampler')
    parser.add_argument('--db_path', type=str, default='data/cifar_origin')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/cifar_original')
    parser.add_argument('--time', type=str, default='uniform')
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--num_imgs', type=int, default=50_000)
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--startbatch', type=int, default=0, help='the batch id to start from')
    args = parser.parse_args()
    generate_samples(args)