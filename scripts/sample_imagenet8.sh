CUDA_VISIBLE_DEVICES=2,0 python3 sampling.py \
--data_dir data/imagenet8 \
--ckpt_path ckpts/imagenet_8 \
--num_imgs 50000 \
--batchsize 200 \
--num_steps 8 \
--dataset imagenet