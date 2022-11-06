import os
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as onp
import jax

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='False'
import flax
import diffusion_distillation
from diffusion_distillation import mydpm
import lmdb

from utils.helper import get_random_label, save2db, gather, load_param


def generate2db(args):
    conditional = False
    if args.dataset == 'cifar10':
        config = diffusion_distillation.config.cifar_distill.get_config()
        x_shape = (32, 32, 3)
    elif args.dataset == 'imagenet':
        config = diffusion_distillation.config.imagenet64_distill.get_config()
        conditional = True
        x_shape = (64, 64, 3)
    else:
        raise ValueError('only cifar10 or imagenet is supported')
    
    model = diffusion_distillation.mymodel.Model(config)
    model.load_teacher_state(args.ckpt_path)
    # replicate to multiple devices
    # model.teacher_state = flax.jax_utils.replicate(model.teacher_state)

    # Function mapping from timestep in [0, 1] to log signal to noise ratio.
    logsnr_schedule_fn = diffusion_distillation.schedules.get_logsnr_schedule(
        config.model.train_logsnr_schedule.name,
        logsnr_min=config.model.train_logsnr_schedule.logsnr_min,
        logsnr_max=config.model.train_logsnr_schedule.logsnr_max,
    )
    loaded_params = load_param(args.ckpt_path, model)
    # define model function

    def model_fn(params, x, logsnr, y=None):
        return model.model.apply(
            {'params': params}, x=x, logsnr=logsnr, y=y, train=False
        )

    # model_fn = lambda x, logsnr, y=None: model.model.apply(
    #     {'params': loaded_params}, x=x, logsnr=logsnr, y=y, train=False)

    # define sampler
    sampler = mydpm.Model(
        model_fn=model_fn,
        mean_type=config.model.mean_type, 
        logvar_type=config.model.logvar_type, 
        logvar_coeff=config.model.get('logvar_coeff', 0.))

    # setup sampling parameters
    num_gpus = jax.local_device_count()
    num_imgs = args.num_imgs
    batchsize = args.batchsize
    local_b = batchsize // num_gpus
    num_batches = num_imgs // batchsize
    sample_key = jax.random.PRNGKey(1)
    
    
    clip_x = config.model.eval_clip_denoised
    # setup database
    os.makedirs(args.data_dir, exist_ok=True)
    db_path = os.path.join(args.data_dir, 'lmdb')
    os.makedirs(db_path, exist_ok=True)

    env = lmdb.open(db_path, map_size=2000*1024*1024*1024, readahead=False)

    # sampling
    label_list = []
    curr = 0
    for batch_id in tqdm(range(num_batches)):
        y_key, x_key, gen_key, sample_key = jax.random.split(sample_key, 4)
        y = get_random_label(num_gpus, local_b, num_class=1000, key=y_key) if conditional else None
        
        init_x = jax.random.normal(x_key, shape=(num_gpus, local_b, *x_shape))
        trajs = sampler.sample_loop(init_x, y, args.num_steps, logsnr_schedule_fn, clip_x)
        # traj_batch = gather(trajs)
        # curr = save2db(traj_batch=traj_batch, env=env, curr=curr)
        if conditional:
            label_list.append(y)
    # write length
    with env.begin(write=True) as txn:
        key = 'length'.encode()
        value = str(curr).encode()
        txn.put(key, value)
    print(f'Write {curr} data to {db_path}')
    if conditional:
        labels = onp.concatenate(label_list, axis=None)
        label_path = os.path.join(args.data_dir, 'labels.npy')
        onp.save(label_path, labels)
        print(f'labels saved to {label_path}')
    print('Complete')


if __name__ == '__main__':
    parser = ArgumentParser(description='parser for DDIM sampler')
    parser.add_argument('--data_dir', type=str, default='data/cifar_origin')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/cifar_original')
    parser.add_argument('--time', type=str, default='uniform')
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--num_imgs', type=int, default=50_000)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--startbatch', type=int, default=0, help='the batch id to start from')
    parser.add_argument('--dataset', type=str, default='cifar10')
    args = parser.parse_args()
    generate2db(args)