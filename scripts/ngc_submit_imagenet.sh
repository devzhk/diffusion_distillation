ngc batch run --name "ml-model.distilled-imagenet-16step" --preempt RUNONCE --commandline  'cd /GAN-code/meta_model/diffusion_distillation; git pull; bash scripts/sample_imagenet16.sh' --image "nvidia/pytorch:22.01-py3" --ace nv-us-west-2 --instance dgx1v.32g.8.norm --result /results --workspace O7-0rdpyTiqLbdKYtM0Lkw:/GAN-code --port 6006 --port 1234 --port 8888

