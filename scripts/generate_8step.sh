pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax ml_collections clu tensorflow clean-fid
python3 sample_distill.py --num_imgs 1024000 --batchsize 4000