import os
from tqdm import tqdm
import lmdb
import numpy as np
import jax.numpy as jnp

from PIL import Image
from cleanfid import fid
import jax
import functools
from utils.helper import download_ckpt


def save2jpg(img, savedir, i):
    save_path = os.path.join(savedir, f'{i}.jpg')
    raw_img = (img + 1.0) * 127.5
    img = np.clip(raw_img, a_min=0, a_max=255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(save_path, quality=100, subsampling=0)


def test_data_load():
    db_path = 'data/cifar8/lmdb'
    env = lmdb.open(db_path, readonly=True, readahead=False, max_readers=128, lock=False, meminit=False)
    C, T, H, W = 3, 9, 32, 32
    traj_shape = (C, T, H, W)
    savedir = 'exp/debug'
    os.makedirs(savedir, exist_ok=True)
    with env.begin(write=False, buffers=True) as txn:
        for i in tqdm(range(50000)):
            key = f'{i}'.encode()
            value = txn.get(key)
            val = np.frombuffer(value, dtype=np.float32)
            data = val.reshape(traj_shape)
            save2jpg(data[:, -1].transpose(1, 2, 0), savedir, i)
    
    score = fid.compute_fid(savedir, dataset_name='cifar10', dataset_res=32, dataset_split='train', mode='legacy_tensorflow')
    print(score)


def test_db():
    traj_shape = (3, 17, 32, 32)
    db_path = 'data/cifar_origin/uniform/lmdb'
    env = lmdb.open(db_path, readonly=True, readahead=False, lock=False, meminit=False)
    with env.begin(write=False, buffers=True) as txn:
        idx = 250 * 1000 + 100
        key = f'{idx}'.encode()
        value = txn.get(key)
        val = np.frombuffer(value, dtype=np.float32)
        data = val.reshape(traj_shape)
        print(data.shape)


def read_db():
    db_path = 'data/cifar8/lmdb'
    env = lmdb.open(db_path, readonly=True, readahead=False)
    with env.begin(write=False) as txn:
        key = 'length'.encode()
        val = int(str(txn.get(key), 'utf-8'))
        print(val)

class Model(object):
    def __init__(self) -> None:
        self.num = 100

    
    def conv(self, x, w, p, j, logsnr_fn):
        output = []
        for i in range(1, len(x) - 1):
            output.append(jnp.dot(x[i-1:i+2], w))
        output = jnp.array(output) * self.num / p + logsnr_fn(j)
        return output


def test_pmap():
    w = np.array([2., 3., 4.])
    model = Model()
    n_devices = jax.local_device_count()
    xs = np.arange(5 * n_devices).reshape(-1, 5)
    ws = np.stack([w] * n_devices)

    logsnr_fn = lambda t: t * 2.0
    conv_fn = jax.pmap(model.conv, axis_name='p', static_broadcasted_argnums=[2, 3, 4])

    res, i, j = conv_fn(xs, ws, 10, 1.0, logsnr_fn)
    print(res, i, j)


def test_bc():
    w = jax.random.normal(jax.random.PRNGKey(1), (4, 5, 3))
    x = 2.0
    out = w * x
    print(out.shape)


def test_for_loop():
    
    def body_fn(i, x):
        jax.debug.print('dtype {x}', x=i.dtype)
        res = x + i
        return res

    init_x = jnp.array([1.0])
    final = jax.lax.fori_loop(0, 10, body_fun=body_fn, init_val=init_x)
    print(final)


def test_download():
    ckpt_path = 'ckpts/imagenet_original'
    if not os.path.exists(ckpt_path):
        download_ckpt(ckpt_path)
    


if __name__ == '__main__':
    # test_data_load()
    # test_db()
    # read_db()
    # test_pmap()
    # test_bc()
    # test_for_loop()
    test_download()