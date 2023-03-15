import os

import numpy as np

from fid import calculate_fid_from_inception_stats, calculate_inception_stats
from generate import db2png

import dnnlib


_ref_dict = {
        'cifar10': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz', 
        'imagenet64': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz'
}
_shape_dict = {
    'cifar10': [3, 17, 32, 32],
    'imagenet64': [3, 9, 64, 64]
}

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""

    print(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    with dnnlib.util.open_url(ref_path) as f:
        ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, 
                                          seed=seed, max_batch_size=batch, num_workers=1)
    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'{fid:g}')
    return fid



def main(num_imgs):
    eval_dict = {
        'cifar10': 'data/cifar_origin/quad/lmdb', 
        'imagenet64': 'data/imagenet16_data/lmdb'
    }
    result = ''
    for key, value in eval_dict.items():
        outdir = os.path.join('figs', key)
        os.makedirs(outdir, exist_ok=True)
        db2png(db_dir=value, outdir=outdir, shape=_shape_dict[key], num_imgs=num_imgs)
        ref_path = _ref_dict[key]
        fid = calc(image_path=outdir, ref_path=ref_path, num_expected=num_imgs, seed=0, batch=64)
        result += f'FID of training set {key} is :{fid:g}\n'
    text_path = '/results/dataset_info.txt'
    with open(text_path, 'w') as f:
        f.write(result)

if __name__ == '__main__':
    main(50000)