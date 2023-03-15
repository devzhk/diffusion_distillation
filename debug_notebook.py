#%%
import os
from argparse import ArgumentParser
from tqdm import tqdm
import pickle

import numpy as onp
import jax
import jax.numpy as jnp
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='False'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
import flax
import diffusion_distillation
from diffusion_distillation import mydpm, mymodel
import lmdb
from PIL import Image

from utils.helper import get_random_label, save2db, gather, load_param, download_ckpt


#%%
def save2dir(images, outdir, curr):
    for i, batch in enumerate(images):
        num_imgs = batch.shape[0]
        batch = (batch + 1.0) * 127.5
        imgs = onp.clip(batch, 0, 255).astype(onp.uint8)
        for j in range(num_imgs):
            im = Image.fromarray(imgs[j])
            img_path = os.path.join(outdir, f'{j}-{i}.png')
            im.save(img_path)    
    return curr + num_imgs

#%%
ckpt_path = 'ckpts/imagenet_16'
base_dir = 'exp/sample16'
img_dir = os.path.join(base_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

x_dir = '../SDE-model/exp/Imagenet-TDDPMm-snr-distill-t4-amp-bx8/random-init'

my_label = 288
seed = 129
pkl_path = os.path.join(x_dir, f'state.pkl')

# ckpt_path = 'ckpts/cifar_original'
# create model 
config = diffusion_distillation.config.imagenet64_distill.get_config()
model = mymodel.Model(config)
# load params
loaded_params = load_param(ckpt_path, model)

logsnr_schedule_fn = diffusion_distillation.schedules.get_logsnr_schedule(
    config.model.train_logsnr_schedule.name,
    logsnr_min=config.model.train_logsnr_schedule.logsnr_min,
    logsnr_max=config.model.train_logsnr_schedule.logsnr_max,
)

def model_fn(params, x, logsnr, y=None):
    return model.model.apply(
        {'params': params}, x=x, logsnr=logsnr, y=y, train=False
    )

# define sampler
sampler = mydpm.Model(
    model_fn=model_fn,
    mean_type=config.model.mean_type, 
    logvar_type=config.model.logvar_type, 
    logvar_coeff=config.model.get('logvar_coeff', 0.))

sample_key = jax.random.PRNGKey(seed)
clip_x = config.model.eval_clip_denoised
#%%
# load init x and label
with open(pkl_path, 'rb') as f:
    state_dict = pickle.load(f)

init_x = state_dict['init_x']
y = state_dict['y']

init_x = jnp.array(init_x.transpose(0, 2, 3, 1))
y = jnp.array(y)

#%%
traj = sampler.sample_loop(loaded_params, init_x, y, num_steps=16, logsnr_schedule_fn=logsnr_schedule_fn, clip_x=clip_x, save_step=1)
# T, B, H, W, C
imgs = jax.device_get(traj)
curr = 0 
curr = save2dir(imgs, img_dir, curr)
# %%
print(init_x.shape)
# %%
