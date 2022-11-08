# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install flax optax ml_collections clu tensorflow

python3 sampling.py \
--data_dir data/imagenet16 \
--ckpt_path ckpts/imagenet_16 \
--num_imgs 2048000 \
--batchsize 640 \
--num_steps 16 \
--dataset imagenet \
--save_step 2