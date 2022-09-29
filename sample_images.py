#%%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='False'
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import jax
import flax
import numpy as onp
import diffusion_distillation
from PIL import Image
from tqdm import tqdm
#%%
loaded_params = diffusion_distillation.checkpoints.restore_from_path('ckpts/cifar_8', target=None)['ema_params']

config = diffusion_distillation.config.cifar_distill.get_config()
model = diffusion_distillation.model.Model(config)
# fix possible flax version errors
ema_params = jax.device_get(model.make_init_state()).ema_params
loaded_params = flax.core.unfreeze(loaded_params)
loaded_params = jax.tree_map(
    lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
    loaded_params,
    flax.core.unfreeze(ema_params))
loaded_params = flax.core.freeze(loaded_params)
del ema_params

batch = {'image': jax.random.normal(jax.random.PRNGKey(10), shape=(250, 32, 32, 3))}
outdir = 'exp/cifar_8'
os.makedirs(outdir, exist_ok=True)

for i in tqdm(range(200)):
    samples = model.samples_fn(rng=jax.random.PRNGKey(i), params=loaded_params, batch=batch, num_samples=250, num_steps=8)
    samples = jax.device_get(samples).astype(onp.uint8)
    num_imgs = samples.shape[0]
    for j in range(num_imgs):
        im = Image.fromarray(samples[j])
        img_path = os.path.join(outdir, f'{i}-{j}.jpg')
        im.save(img_path, quality=100, subsampling=0)
# %%
