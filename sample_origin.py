import os
import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as onp
import jax
import jax.numpy as jnp
import tensorflow as tf
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='False'
import flax
import flax.linen as nn
from PIL import Image
import diffusion_distillation
from diffusion_distillation import utils
import lmdb


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


def load_teacher(args, config):
    # Create teacher model.
    teacher = diffusion_distillation.model.Model(config)

    # Load the teacher params.
    cache_dir = os.path.join(os.environ["HOME"], ".cache/flax_diffusion")
    os.makedirs(cache_dir, exist_ok=True)
    if args.ckpt_path.startswith('gs://'):
        local_teacher_path = os.path.join(
            cache_dir, os.path.basename(args.ckpt_path))
        if not os.path.isfile(local_teacher_path) and args.dist.rank == 0:
            tf.io.gfile.copy(args.ckpt_path, local_teacher_path)
        while not os.path.isfile(local_teacher_path):
            # Wait for the master to copy the file.
            print('Rank', args.dist.rank, 'waiting for teacher params to download to',
                  local_teacher_path, '...')
            time.sleep(1)
    else:
        local_teacher_path = args.ckpt_path
    teacher.load_teacher_state(local_teacher_path)

    return teacher


def evaluate_teacher(args):
    config = diffusion_distillation.config.cifar_distill.get_config()
    teacher = load_teacher(args, config)
    teacher.teacher_state = flax.jax_utils.replicate(teacher.teacher_state)

    # Function mapping from timestep in [0, 1] to log signal to noise ratio.
    logsnr_schedule_fn = diffusion_distillation.schedules.get_logsnr_schedule(
        teacher.config.model.train_logsnr_schedule.name,
        logsnr_min=teacher.config.model.train_logsnr_schedule.logsnr_min,
        logsnr_max=teacher.config.model.train_logsnr_schedule.logsnr_max,
    )

    def teacher_fn(params, xt, logsnr, y=None):
        with jax.default_matmul_precision("bfloat16"):
            return teacher.model.apply(
                {'params': params}, x=xt, logsnr=logsnr, y=y, train=False)

    def ddim_step_fn(teacher_params, xt, y, t, s):
        logsnr_t = logsnr_schedule_fn(t)
        xhat = teacher_fn(
            params=teacher_params,
            xt=xt,
            logsnr=logsnr_t,
            y=y,
        )

        bc = diffusion_distillation.utils.broadcast_from_left
        logsnr_t = bc(logsnr_t, xt.shape)
        alpha_t = jnp.sqrt(nn.sigmoid(logsnr_t))
        sigma_t = jnp.sqrt(nn.sigmoid(-logsnr_t))
        epshat = (xt - alpha_t * xhat) / sigma_t

        logsnr_s = logsnr_schedule_fn(s)
        logsnr_s = bc(logsnr_s, xt.shape)
        alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
        sigma_s = jnp.sqrt(nn.sigmoid(-logsnr_s))
        xt = alpha_s * xhat + sigma_s * epshat
        return xhat, xt

    ddim_step_fn_p = jax.pmap(ddim_step_fn, axis_name='batch')
    assert teacher.config.model.mean_type == 'x'

    def sample_ddim(rng, teacher_params, z, y, num_steps, t_idx):
        dt = 1. / num_steps
        xt = z
        x_list = [xt]
        for i, timestep in enumerate(jnp.linspace(1., dt, num_steps)):
            t = jnp.full(z.shape[:2], fill_value=timestep, dtype=jnp.float32)
            s = t - dt
            xhat, xt = ddim_step_fn_p(teacher_params, xt, y, t, s)
            if i in t_idx:
                x_list.append(xhat)
        return x_list

    num_gpus = jax.local_device_count()
    # Generate N images.
    N = args.num_imgs
    B = args.batchsize
    local_b = B // num_gpus

    num_batches = N // B - args.startbatch
    sample_key = jax.random.PRNGKey(123)
    z1_shape = (32, 32, 3)
    num_steps = args.num_steps
    curr = B * args.startbatch
    db_path = os.path.join(args.db_path, args.time, 'lmdb')
    os.makedirs(db_path, exist_ok=True)
    env = lmdb.open(db_path, map_size=300*1024*1024*1024, readahead=False)
    assert num_steps == 512
    if args.time == 'quad':
        t_idx = [num_steps - 1 - 2 * i * i for i in range(17)]
    else:
        step = num_steps // 16
        t_idx = [step * i - 1 for i in range(17)]
    for batch_idx in tqdm(range(num_batches)):
        y_key, z_key, gen_key, sample_key = jax.random.split(sample_key, 4)
        # y = sample_labels(
        #     y_key, num_classes=teacher.model.num_classes, num=B,
        #     limit_classes=args.distillation.classes)
        y = None
        # if y is not None:
        #     assert y.shape == (B,)
        #     y = y.reshape(
        #         args.dist.local_device_count, local_b)

        z1 = jax.random.normal(
            z_key, shape=(num_gpus, local_b, *z1_shape))
        traj_list = sample_ddim(
            jax.random.split(gen_key, num_gpus),
            teacher.teacher_state.ema_params,
            z1, y, num_steps=num_steps, t_idx=t_idx)
        # T, num_gpus, local_b, H, W, C
        # images_batch = images_batch.reshape(B, *images_batch.shape[2:])
        # images_batch= jnp.clip(utils.unnormalize_data(images_batch), 0, 255)
        # Save a grid of samples used for FID computation.
        traj_list = jax.device_get(traj_list)
        traj_batch = onp.stack(traj_list, axis=2)   # B, T, H, W, C
        traj_batch = traj_batch.reshape(B, *traj_batch.shape[2:])
        traj_batch = traj_batch.transpose(0, 4, 1, 2, 3)  # B, C, T, H, W
        traj_batch = onp.ascontiguousarray(traj_batch)
        curr = save2db(traj_batch=traj_batch, env=env, curr=curr)

    with env.begin(write=True) as txn:
        key = 'length'.encode()
        value = str(curr).encode()
        txn.put(key, value)

    print(f'Write {curr} data to {args.db_path}')


if __name__ == '__main__':
    parser = ArgumentParser(description='parser for DDIM sampler')
    parser.add_argument('--db_path', type=str, default='data/cifar_origin')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/cifar_original')
    parser.add_argument('--time', type=str, default='uniform')
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--num_imgs', type=int, default=16_000)
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--startbatch', type=int, default=0, help='the batch id to start from')
    args = parser.parse_args()
    evaluate_teacher(args)
