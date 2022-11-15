import os
from tqdm import tqdm
from argparse import ArgumentParser
import lmdb
import numpy as np
import shutil


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
        for key, value in tqdm(cursor):
            curr_key = f'{curr}'.encode()
            target_txn.put(curr_key, value)
            curr += 1


def process_fn(args):
    sub_folders = [os.path.join(f.path, 'lmdb') for f in os.scandir(args.datadir) if f.is_dir()]
    db_folders = list(filter(check_db, sub_folders))
    
    target_db = 'data/imagenet16_db/lmdb'
    target_label_path = target_db.replace('lmdb', 'labels.npy')
    target_labels = np.load(target_label_path)
    log_file = open(args.logpath, 'w')
    print(db_folders, file=log_file)
    target_env = lmdb.open(target_db, 
                           map_size=1200*1024*1024*1024, 
                           map_async=True, writemap=True, readahead=False)
    for db_dir in db_folders:
        print(db_dir)
        # merge dbs
        with lmdb.open(db_dir, map_size=1200*1024*1024*1024, 
                       map_async=True, writemap=True, readahead=False) as source_env:
            merge_db(target_env, source_env)
        # merge labels 
        label_path = db_dir.replace('lmdb', 'labels.npy')
        labels = np.load(label_path)
        target_labels = np.concatenate((target_labels, labels), axis=None)
    np.save(target_label_path, target_labels)
    print(f'labels saved to {target_label_path}', file=log_file)
    num_imgs = target_env.stat()['entries']
    print(f'{num_imgs} in the whole database', file=log_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--logpath', type=str, default='/results/merge_db.txt')
    args = parser.parse_args()
    process_fn(args)
