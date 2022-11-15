import os
from tqdm import tqdm
from argparse import ArgumentParser
import lmdb
import numpy as np
import shutil

import jax

seed_dict = {
    'data/imagenet16_db/lmdb': 12, 
    'data/imagenet16_db1/lmdb': 132, 
    'data/imagenet16_db2/lmdb': 133, 
    'data/imagenet16_db3/lmdb': 134,
    'data/imagenet16_db4/lmdb': 135, 
    'data/imagenet16_db5/lmdb': 136, 
    'data/imagenet16_db6/lmdb': 236, 
    'data/imagenet16_db7/lmdb': 237, 
    'data/imagenet16_db8/lmdb': 238,
    'data/imagenet16_db9/lmdb': 239, 
    'data/imagenet16_db10/lmdb': 336, 
    'data/imagenet16_db11/lmdb': 337, 
    'data/imagenet16_db12/lmdb': 338, 
    'data/imagenet16_db13/lmdb': 339, 
}


def delete_db(db_dir):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)


def test_os():
    sub_folders = [f.path for f in os.scandir('data') if f.is_dir()]
    print(sub_folders)


def check_db(db_path):
    if 'imagenet_db' in db_path and db_path != 'data/imagenet16_db/lmdb':
        return True
    else:
        return False


def merge_db(target_env, source_env):
    curr = target_env.stat()['entries']
    source_txn = source_env.begin(write=False)
    with target_env.begin(write=True) as target_txn:
        cursor = source_txn.cursor()
        for key, value in cursor:
            curr_key = f'{curr}'.encode()
            target_txn.put(curr_key, value)
            curr += 1


def get_num_imgs(db_dir) -> int:
    env = lmdb.open(db_dir)
    num_imgs = env.stat()['entries']
    env.close()
    return num_imgs


def generate_labels():
    # recover setup
    num_gpus = 8
    batchsize = 2048
    local_b = batchsize // num_gpus
    
    db_list = []
    yarr_list = []
    for key, value in seed_dict.items():        
        sample_key = jax.random.PRNGKey(value)
        num_imgs = get_num_imgs(key)
        if num_imgs % batchsize == 0:
            num_batches = num_imgs // batchsize
        else:
            raise ValueError('number of entries is not divisible by batchsize!')
        
        label_arr = np.zeros(num_imgs, dtype=np.int32)
        for batch_id in range(num_batches):
            y_key, x_key, gen_key, sample_key = jax.random.split(sample_key, 4)
            y = jax.random.randint(y_key, (num_gpus, local_b, ), minval=0, maxval=1000)
            labels = jax.device_get(y).reshape(-1)
            label_arr[batch_id * batchsize: (batch_id + 1) * batchsize] = labels
        yarr_list.append(label_arr)
        # add to db list 
        if value != 12:
            db_list.append(key)
    all_labels = np.concatenate(yarr_list, axis=None)
    label_path = 'data/imagenet16_db/labels.npy'
    np.save(label_path, all_labels)
    return db_list


def process_fn(args):
    db_folders = generate_labels()
    
    target_db = 'data/imagenet16_db/lmdb'
    log_file = open(args.logpath, 'w')
    print(db_folders, file=log_file)
    target_env = lmdb.open(target_db, 
                           map_size=1200*1024*1024*1024, 
                           map_async=True, writemap=True, readahead=False)
    for db_dir in db_folders:
        print(db_dir, file=log_file)
        # merge dbs
        with lmdb.open(db_dir, map_size=1200*1024*1024*1024, 
                       map_async=True, writemap=True, readahead=False) as source_env:
            merge_db(target_env, source_env)
    num_imgs = target_env.stat()['entries']
    print(f'{num_imgs} in the whole database', file=log_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--logpath', type=str, default='/results/merge_db.txt')
    args = parser.parse_args()
    process_fn(args)
    # generate_labels()
