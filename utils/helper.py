import os
from tqdm import tqdm
import numpy as onp
import jax
import jax.numpy as jnp
from PIL import Image
import flax
from flax import serialization
import requests


def get_random_label(num_gpu, batch, num_class, key):
    '''
    sample random class labels. 
    '''
    labels = jax.random.randint(key, (num_gpu, batch, ), minval=0, maxval=num_class, dtype=jnp.int32)
    return labels


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


def save2dir(images, outdir, curr):
    num_imgs = images.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(images[j])
        img_path = os.path.join(outdir, f'{j + curr}.jpg')
        im.save(img_path, quality=100, subsampling=0)    
    return curr + num_imgs


def load_param(path, model):
    '''
    load params from path 
    '''
    with open(path, 'rb') as f:
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
    return loaded_params


def gather(arr_list):
    '''
    Args:
        - arr_list: list of shared array, T x (num_gpus, local_b, H, W, C)
    Return:
        - arr_batch: numpy array, (batchsize, C, T, H, W)
    '''
    arr_list = jax.device_get(arr_list)
    arr_batch = onp.stack(arr_list, axis=2) # (num_gpus, local_b, T, H, W, C)
    arr_batch = arr_batch.reshape(-1, *arr_batch.shape[2:])
    arr_batch = arr_batch.transpose(0, 4, 1, 2, 3)  # (batchsize, C, T, H, W)
    arr_batch = onp.ascontiguousarray(arr_batch)
    return arr_batch


def download_ckpt(file_path):
    url_dict = {
        'imagenet_16': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/imagenet_16', 
        'imagenet_8': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/imagenet_8', 
        'imagenet_original': 'https://hkzdata.s3.us-west-2.amazonaws.com/SBM/diffusion_distillation/imagenet_original'
    }
    if 'imagenet_16' in file_path:
        url = url_dict['imagenet_16']
    elif 'imagenet_8' in file_path:
        url = url_dict['imagenet_8']
    elif 'imagenet_original' in file_path:
        url = url_dict['imagenet_original']
    
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024 * 1024)):
                f.write(chunk)
    print('Complete')