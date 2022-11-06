"""My diffusion implementation for sampling trajectories"""

# pylint:disable=missing-class-docstring,missing-function-docstring
# pylint:disable=logging-format-interpolation
# pylint:disable=g-long-lambda

from . import utils
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp

### Basic diffusion process utilities


def diffusion_reverse(*, x, z_t, logsnr_s, logsnr_t, x_logvar):
  """q(z_s | z_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
  alpha_st = jnp.sqrt((1. + jnp.exp(-logsnr_t)) / (1. + jnp.exp(-logsnr_s)))
  alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
  r = jnp.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
  one_minus_r = -jnp.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
  log_one_minus_r = utils.log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))

  mean = r * alpha_st * z_t + one_minus_r * alpha_s * x

  if isinstance(x_logvar, str):
    if x_logvar == 'small':
      # same as setting x_logvar to -infinity
      var = one_minus_r * nn.sigmoid(-logsnr_s)
      logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
    elif x_logvar == 'large':
      # same as setting x_logvar to nn.log_sigmoid(-logsnr_t)
      var = one_minus_r * nn.sigmoid(-logsnr_t)
      logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_t)
    elif x_logvar.startswith('medium:'):
      _, frac = x_logvar.split(':')
      frac = float(frac)
      logging.info('logvar frac=%f', frac)
      assert 0 <= frac <= 1
      min_logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
      max_logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_t)
      logvar = frac * max_logvar + (1 - frac) * min_logvar
      var = jnp.exp(logvar)
    else:
      raise NotImplementedError(x_logvar)
  else:
    assert isinstance(x_logvar, jnp.ndarray) or isinstance(
        x_logvar, onp.ndarray)
    assert x_logvar.shape == x.shape
    # start with "small" variance
    var = one_minus_r * nn.sigmoid(-logsnr_s)
    logvar = log_one_minus_r + nn.log_sigmoid(-logsnr_s)
    # extra variance weight is (one_minus_r*alpha_s)**2
    var += jnp.square(one_minus_r) * nn.sigmoid(logsnr_s) * jnp.exp(x_logvar)
    logvar = jnp.logaddexp(
        logvar, 2. * log_one_minus_r + nn.log_sigmoid(logsnr_s) + x_logvar)
  return {'mean': mean, 'std': jnp.sqrt(var), 'var': var, 'logvar': logvar}


def diffusion_forward(*, x, logsnr):
  """q(z_t | x)."""
  return {
      'mean': x * jnp.sqrt(nn.sigmoid(logsnr)),
      'std': jnp.sqrt(nn.sigmoid(-logsnr)),
      'var': nn.sigmoid(-logsnr),
      'logvar': nn.log_sigmoid(-logsnr)
  }


def predict_x_from_eps(*, z, eps, logsnr):
  """x = (z - sigma*eps)/alpha."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  return jnp.sqrt(1. + jnp.exp(-logsnr)) * (
      z - eps * jax.lax.rsqrt(1. + jnp.exp(logsnr)))


def predict_xlogvar_from_epslogvar(*, eps_logvar, logsnr):
  """Scale Var[eps] by (1+exp(-logsnr)) / (1+exp(logsnr)) = exp(-logsnr)."""
  return eps_logvar - logsnr


def predict_eps_from_x(*, z, x, logsnr):
  """eps = (z - alpha*x)/sigma."""
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  return jnp.sqrt(1. + jnp.exp(logsnr)) * (
      z - x * jax.lax.rsqrt(1. + jnp.exp(-logsnr)))


def predict_epslogvar_from_xlogvar(*, x_logvar, logsnr):
  """Scale Var[x] by (1+exp(logsnr)) / (1+exp(-logsnr)) = exp(logsnr)."""
  return x_logvar + logsnr


def predict_x_from_v(*, z, v, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, z.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * z - sigma_t * v


def predict_v_from_x_and_eps(*, x, eps, logsnr):
  logsnr = utils.broadcast_from_left(logsnr, x.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * eps - sigma_t * x


class Model:

  def __init__(self, model_fn, *, mean_type, logvar_type, logvar_coeff,
               target_model_fn=None):
    self.model_fn = model_fn
    self.mean_type = mean_type
    self.logvar_type = logvar_type
    self.logvar_coeff = logvar_coeff
    self.target_model_fn = target_model_fn

  def _run_model(self, *, z, y, logsnr, model_fn, clip_x):
    model_output = model_fn(z, logsnr, y)
    if self.mean_type == 'eps':
      model_eps = model_output
    elif self.mean_type == 'x':
      model_x = model_output
    elif self.mean_type == 'v':
      model_v = model_output
    elif self.mean_type == 'both':
      _model_x, _model_eps = jnp.split(model_output, 2, axis=-1)  # pylint: disable=invalid-name
    else:
      raise NotImplementedError(self.mean_type)

    # get prediction of x at t=0
    if self.mean_type == 'both':
      # reconcile the two predictions
      model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
      wx = utils.broadcast_from_left(nn.sigmoid(-logsnr), z.shape)
      model_x = wx * _model_x + (1. - wx) * model_x_eps
    elif self.mean_type == 'eps':
      model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
    elif self.mean_type == 'v':
      model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

    # clipping
    if clip_x:
      model_x = jnp.clip(model_x, -1., 1.)

    # get eps prediction if clipping or if mean_type != eps
    if self.mean_type != 'eps' or clip_x:
      model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

    # get v prediction if clipping or if mean_type != v
    if self.mean_type != 'v' or clip_x:
      model_v = predict_v_from_x_and_eps(
          x=model_x, eps=model_eps, logsnr=logsnr)

    return {'model_x': model_x,
            'model_eps': model_eps,
            'model_v': model_v}

  def ddim_step(self, i, z_t, y, num_steps, logsnr_schedule_fn, clip_x):
    shape, dtype = z_t.shape, z_t.dtype
    i = jnp.array(i, dtype=dtype)
    logsnr_t = logsnr_schedule_fn((i + 1.)/ num_steps)
    logsnr_s = logsnr_schedule_fn(i / num_steps)
    model_out = self._run_model(
        z=z_t,
        y=y,
        logsnr=jnp.full((shape[0],), logsnr_t),
        model_fn=self.model_fn,
        clip_x=clip_x)
    x_pred_t = model_out['model_x']
    eps_pred_t = model_out['model_eps']
    stdv_s = jnp.sqrt(nn.sigmoid(-logsnr_s))
    alpha_s = jnp.sqrt(nn.sigmoid(logsnr_s))
    z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
    return x_pred_t, z_s_pred

  def sample_loop(self, init_x, y, num_steps,
                  logsnr_schedule_fn, clip_x):
    '''
    Args:
      - init_x: (num_gpus, local_b, *x_shape)
      - y: (num_gpus, local_b)
    Return:
      traj: list of (num_gpus, local_b, *x_shape)
    '''    

    ddim_step_fn = jax.pmap(self.ddim_step, axis_name='p', 
                            static_broadcasted_argnums=[0, 3, 4, 5])

    zt = init_x
    x_list = [init_x]
    # loop over t = num_steps-1, ..., 0
    for i in reversed(range(num_steps)):
      xhat, zt = ddim_step_fn(i, zt, y, num_steps, logsnr_schedule_fn, clip_x)
      x_list.append(xhat)
    return x_list
