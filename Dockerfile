FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN pip install wandb tqdm pyyaml 
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install chex flax optax ml_collections clu tensorflow
RUN pip install clean-fid h5py