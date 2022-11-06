CUDA_VISIBLE_DEVICES=0,1,2,3 python3 sampling.py \
--data_dir data/imagenet8 \
--ckpt_path ckpts/imagenet_8 \
--num_imgs 50000 \
--batchsize 400 \
--num_steps 8 \
--dataset imagenet