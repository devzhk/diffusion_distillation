# Progressive Distillation for Fast Sampling of Diffusion Models
This repo samples DDIM trajectory using code and model checkpoints of the <a href="https://openreview.net/forum?id=TIdIXIpzhoI">ICLR 2022 paper</a> by Tim Salimans and Jonathan Ho. 

- Code is adapted to latest jax and flax. 
- Sampling fn is adapted for trajectory sampling. 
<a href="https://colab.research.google.com/github/google-research/google-research/blob/master/diffusion_distillation/diffusion_distillation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

![FID vs number of steps](fid_steps_graph.png)
