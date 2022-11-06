import functools
from typing import Any, Dict, Union

from . import checkpoints
from . import datasets
from . import mydpm
from . import schedules
from . import unet
from . import utils
from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as onp


@flax.struct.dataclass
class TrainState:
  step: int
  ema_params: Any
  num_sample_steps: int


class Model:
  """Diffusion model."""

  def __init__(self, config, dataset=None):
    self.config = config

    if dataset is not None:
      self.dataset = dataset
    else:
      self.dataset = getattr(datasets, config.dataset.name)(
          **config.dataset.args)

    self._eval_step = None

    # infer number of output channels for UNet
    x_ch = self.dataset.data_shape[-1]
    out_ch = x_ch
    if config.model.mean_type == 'both':
      out_ch += x_ch
    if 'learned' in config.model.logvar_type:
      out_ch += x_ch

    self.model = unet.UNet(
        num_classes=self.dataset.num_classes,
        out_ch=out_ch,
        **config.model.args)

  @property
  def current_num_steps(self):
    if hasattr(self.config, 'distillation'):
      assert hasattr(self, 'teacher_state')
      return int(self.teacher_state.num_sample_steps // 2)
    else:
      return self.config.model.train_num_steps

  def make_init_params(self, global_rng):
    init_kwargs = dict(
        x=jnp.zeros((1, *self.dataset.data_shape), dtype=jnp.float32),
        y=jnp.zeros((1,), dtype=jnp.int32),
        logsnr=jnp.zeros((1,), dtype=jnp.float32),
        train=False,
    )
    return self.model.init({'params': global_rng}, **init_kwargs)['params']

  def make_init_state(self):
    """Make an initial TrainState."""
    # Init model params (same rng across hosts)
    init_params = self.make_init_params(
        global_rng=jax.random.PRNGKey(self.config.seed))
    logging.info('Param shapes: {}'.format(
        jax.tree_map(lambda a: a.shape, init_params)))
    logging.info('Number of trainable parameters: {:,}'.format(
        utils.count_params(init_params)))

    # For ema_params below, copy so that pmap buffer donation doesn't donate the
    # same buffer twice
    return TrainState(
        step=0,
        ema_params=utils.copy_pytree(init_params),
        num_sample_steps=self.config.model.train_num_steps)

  def load_teacher_state(self, ckpt_path=None):
    """Load teacher state and fix flax version incompatibilities."""
    teacher_state = jax.device_get(
        self.make_init_state())
    if ckpt_path is None:
      ckpt_path = self.config.distillation.teacher_checkpoint_path
    loaded_state = checkpoints.restore_from_path(ckpt_path, target=None)
    teacher_params = loaded_state['ema_params']
    teacher_params = flax.core.unfreeze(teacher_params)
    teacher_params = jax.tree_map(
        lambda x, y: onp.reshape(x, y.shape) if hasattr(y, 'shape') else x,
        teacher_params,
        flax.core.unfreeze(teacher_state.ema_params))
    teacher_params = flax.core.freeze(teacher_params)
    if ('num_sample_steps' in loaded_state and
        loaded_state['num_sample_steps'] > 0):
      num_sample_steps = loaded_state['num_sample_steps']
    else:
      num_sample_steps = self.config.distillation.start_num_steps
    self.teacher_state = TrainState(
        step=0,  # reset number of steps
        ema_params=teacher_params,
        num_sample_steps=num_sample_steps,
        )

  def samples_fn(self,
                 *,
                 rng,
                 params,
                 init_x,
                 labels=None,
                 num_steps=None):
    """Sample from the model."""
    rng = utils.RngGen(rng)
    y = labels
    
    model_fn = lambda x, logsnr: self.model.apply(
        {'params': params}, x=x, logsnr=logsnr, y=y, train=False)

    if num_steps is None:
      num_steps = self.config.model.eval_sampling_num_steps
    logging.info(
        f'eval_sampling_num_steps: {num_steps}'
    )
    logging.info(
        f'eval_logsnr_schedule: {self.config.model.eval_logsnr_schedule}')

    model = mydpm.Model(
        model_fn=model_fn,
        mean_type=self.config.model.mean_type,
        logvar_type=self.config.model.logvar_type,
        logvar_coeff=self.config.model.get('logvar_coeff', 0.))
    samples = model.sample_loop(
        rng=next(rng),
        init_x=init_x,
        num_steps=num_steps,
        logsnr_schedule_fn=schedules.get_logsnr_schedule(
            **self.config.model.eval_logsnr_schedule),
        sampler=self.config.sampler,
        clip_x=self.config.model.eval_clip_denoised)

    unnormalized_samples = jnp.clip(utils.unnormalize_data(samples), 0, 255)
    return unnormalized_samples
