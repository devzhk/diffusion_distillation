#%%
import jax
import flax
from flax import serialization
import numpy as onp
import jax.numpy as jnp
import diffusion_distillation
from diffusion_distillation import dpm
from diffusion_distillation import schedules

#%%
ckpt_path = 'ckpts/imagenet_original'
# create model 
config = diffusion_distillation.config.imagenet64_base.get_config()
teacher = diffusion_distillation.model.Model(config)

# load params
with open(ckpt_path, 'rb') as f:
    loaded_params = serialization.from_bytes(target=None, encoded_bytes=f.read())['ema_params']

# fix possible flax version errors
ema_params = jax.device_get(teacher.make_init_state()).ema_params
loaded_params = flax.core.unfreeze(loaded_params)
loaded_params = jax.tree_map(
    lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
loaded_params,
flax.core.unfreeze(ema_params))
loaded_params = flax.core.freeze(loaded_params)
del ema_params

#%%
def modelfn(params, x, logsnr, y=None):
    return teacher.model.apply(
        {'params': params}, x=x, logsnr=logsnr, y=y, train=False
    )

# 
model = dpm.Model(
    model_fn=modelfn, 
    mean_type=config.model.mean_type, 
    logvar_type=config.model.logvar_type, 
    logvar_coeff=config.model.get('logvar_coeff', 0.))

# prepare model
logsnr_fn = schedules.get_logsnr_schedule(**config.model.eval_logsnr_schedule)


# create random init_x
x_shape = (64, 64, 3)
B = 2
init_path = 'exp/init_x.npy'
out_path = 'exp/out-jax.npy'
init_x = onp.random.normal(size=(B, *x_shape))
onp.save(init_path, init_x)

x = jnp.asarray(init_x)
print(x.dtype)
labels = 867 * jnp.ones((B, ), dtype=jnp.int32)

# define model function 
model_fn = lambda x, logsnr: teacher.model.apply(
    {'params': loaded_params}, x=x, logsnr=logsnr, y=labels, train=False
)


logsnr_t = logsnr_fn(0.5)
logsnr = jnp.full((B, ), logsnr_t)

clip_x = False

model_output = model._run_model(z=x, logsnr=logsnr, model_fn=model_fn, clip_x=clip_x)

output = jax.device_get(model_output)
onp.save(out_path, output)