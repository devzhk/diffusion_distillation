ngc batch run --name "ml-model.dsno-exempt-eval-datasets" \
--preempt RUNONCE \
--commandline  'cd /GAN-code/meta_model/diffusion_distillation; git config --global --add safe.directory /GAN-code/meta_model/diffusion_distillation; git pull; python3 eval_data.py' --image "nvidia/pytorch:22.01-py3" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --result /results --workspace O7-0rdpyTiqLbdKYtM0Lkw:/GAN-code --port 6006 --port 1234 --port 8888


