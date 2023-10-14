# %%
# Based on https://github.com/lucidrains/denoising-diffusion-pytorch

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import yaml, os


data_path = './data/BRICKS_colorful_final'
results_path = './results_colorful_final'
checkpoint_num = '11'        # Checkpoint to load from, set to None if starting from scratch

# path = './data/vae'
# results_path = './results_vae_128'
# checkpoint_num = None

# Unet parameters
dim = 64
dim_mults = (1, 2, 4, 8)
flash_attn = True

# Gauusian Diffusion model parameters
image_size = 64
timesteps = 1000
sampling_timesteps = 250

# Trainer parameters
train_batch_size = 32
train_lr = 8e-5
train_num_steps = 20_000         # total training steps
gradient_accumulate_every = 2    # gradient accumulation steps
ema_decay = 0.995                # exponential moving average decay
amp = True                       # turn on mixed precision
calculate_fid = False # whether to calculate fid during training


with open(os.path.join(results_path, 'config.yaml'), "w") as f:
    f.write(yaml.dump(
        {
            "unet": {"dim": dim, "dim_mults": dim_mults, "flash_attn": flash_attn},
            "diffusion": {"image_size": image_size, "timesteps": timesteps, "sampling_timesteps": sampling_timesteps},
            "trainer": {"train_batch_size": train_batch_size, "train_lr": train_lr, "train_num_steps": train_num_steps, "gradient_accumulate_every": gradient_accumulate_every, "ema_decay": ema_decay, "amp": amp, "calculate_fid": calculate_fid, "results_folder": results_path, "data_path": data_path},
        }
    ))
# %%

model = Unet(
    dim = dim,
    dim_mults = dim_mults,
    flash_attn = flash_attn 
)

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = timesteps,                  # number of steps
    sampling_timesteps = sampling_timesteps # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size = train_batch_size,
    train_lr = train_lr,
    train_num_steps = train_num_steps,         # total training steps
    gradient_accumulate_every = gradient_accumulate_every,    # gradient accumulation steps
    ema_decay = ema_decay,                # exponential moving average decay
    amp = amp,                       # turn on mixed precision
    calculate_fid = calculate_fid, # whether to calculate fid during training
    results_folder = results_path      # folder to save results
)

if checkpoint_num:
    trainer.load(checkpoint_num)

trainer.train()