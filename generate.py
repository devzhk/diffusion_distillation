import os
from tqdm import tqdm
import numpy as onp
from argparse import ArgumentParser
import lmdb

from diffusion_distillation.utils import unnormalize_data
from utils.helper import save2dir
from PIL import Image


def readdb(txn, i, shape):
    key = f'{i}'.encode()
    value = txn.get(key)
    value = onp.frombuffer(value, dtype=onp.float32)
    val = value.reshape(shape)[:, -1, :, :] # C, H, W
    data = val.transpose([1, 2, 0])
    return data


def db2jpg(args):
    '''
    convert images from database to jpgs
    '''
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    env = lmdb.open(args.db_dir, readonly=True, readahead=False, meminit=False)
    
    if 'cifar' in args.db_dir:
        shape = [3, 9, 32, 32]
    else:
        shape = [3, 9, 64, 64]

    with env.begin(write=False) as txn:
        N = int(str(txn.get('length'.encode()), 'utf-8'))
        for i in tqdm(range(N)):
            img = readdb(txn, i, shape)
            img = onp.clip(unnormalize_data(img), 0, 255).astype(onp.uint8)
            im = Image.fromarray(img)
            img_path = os.path.join(outdir, f'{i}.jpg')
            im.save(img_path, quality=100, subsampling=0)
    print('Done')



if __name__ == '__main__':
    parser = ArgumentParser(description='parser for DDIM sampler')
    parser.add_argument('--db_dir', type=str, default='data/imagenet8/lmdb')
    parser.add_argument('--outdir', type=str, default='exp/imagenet8_db')
    args = parser.parse_args()
    db2jpg(args)