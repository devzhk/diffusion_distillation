# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install flax optax ml_collections clu tensorflow
python3 sampling.py \
--data_dir data/imagenet16_db10 \
--ckpt_path ckpts/imagenet_16 \
--num_imgs 204800 \
--batchsize 2048 \
--num_steps 16 \
--dataset imagenet \
--save_step 2 \
--seed 336 \
--startbatch 0