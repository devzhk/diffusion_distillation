CUDA_VISIBLE_DEVICES=0,1 python3 sampling.py \
--data_dir data/imagenet16 \
--ckpt_path ckpts/imagenet_16 \
--num_imgs 102400 \
--batchsize 160 \
--num_steps 16 \
--dataset imagenet \
--save_step 2