# Based on https://github.com/lucidrains/denoising-diffusion-pytorch
# %%
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


# path = './data/BRICKS_colorful_final'
# results_path = './results_colorful_final'
# checkpoint_num = '11'

path = './data/vae'
results_path = './results_vae_128'
checkpoint_num = None

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 2000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False, # whether to calculate fid during training
    results_folder = results_path      # folder to save results
)

if checkpoint_num:
    trainer.load(checkpoint_num)

trainer.train()