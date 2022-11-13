import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from . import layers
from . import layersm

get_act = layers.get_act
default_initializer = layers.default_init
conv3x3 = layers.ddpm_conv3x3
Downsample = layers.Downsample
Upsample = layers.Upsample
NIN = layers.NIN
MultiheadAttn = layersm.MultiheadAttnBlock
ResidualBlockm = layersm.ResidualBlockm


def get_logsnr_input(logsnr, logsnr_type='inv_cos'):
    if logsnr_type == 'inv_cos':
        logsnr_input = torch.atan(torch.exp(- 0.5 * torch.clamp(logsnr, min=-20., max=20.))) / (0.5 * np.pi)
    elif logsnr_type == 'sigmoid':
        logsnr_input = torch.sigmoid(logsnr)
    else:
        raise ValueError(f'{logsnr_type} not supported')
    return logsnr_input


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    timesteps *= (1000.0 / max_time)
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(- torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb



class DDPMm(nn.Module):
    '''
    Pytorch implementation of the architecture used in progressive knowledge distillation.
    Adapted from Jax code of https://github.com/google-research/google-research/blob/master/diffusion_distillation/diffusion_distillation/unet.py
    '''
    def __init__(self, config):
        super(DDPMm, self).__init__()
        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        self.temb_dim = temb_dim = config.model.temb_dim
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        self.num_attn_heads = num_attn_heads = config.model.num_attn_heads
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        self.logsnr_type = config.model.logsnr_type
        channels = config.data.num_channels
        init_scale = config.model.init_scale

        ResnetBlock = partial(ResidualBlockm,
                              act=act,
                              dropout=dropout,
                              init_scale=init_scale,
                              temb_dim=temb_dim)
        AttnBlock = partial(MultiheadAttn,
                            num_heads=num_attn_heads)

        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, temb_dim)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(temb_dim, temb_dim))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        # downsample part
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch, with_conv=resamp_with_conv))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, down=True))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # upsample part
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))
        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, labels):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            logsnr_input = get_logsnr_input(labels, logsnr_type=self.logsnr_type)
            temb = get_timestep_embedding(logsnr_input, self.nf, max_time=1.0)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None
        h = x
        # Downsampling block
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                hs.append(h)
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        return h