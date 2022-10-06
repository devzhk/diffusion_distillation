import os
from tqdm import tqdm
import lmdb
import numpy as np

from PIL import Image
from cleanfid import fid


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


if __name__ == '__main__':
    # test_data_load()
    test_db()