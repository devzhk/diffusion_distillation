import os
from argparse import ArgumentParser
import requests
from tqdm import tqdm


url_dict = {
    'openai-ref': 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz', 
    'EDM-ref': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64-baseline.npz'
}


def download_file(url, file_path):
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024 * 1024)):
                f.write(chunk)
    print('Complete')


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for downloading data')
    parser.add_argument('--name', type=str, default='EDM-ref')
    parser.add_argument('--outdir', type=str, default='data')
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    file_path = os.path.join(args.outdir, f'{args.name}.npz')

    download_url = url_dict[args.name]
    
    download_file(download_url, file_path)