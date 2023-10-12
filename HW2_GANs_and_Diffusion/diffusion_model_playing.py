# Based on https://github.com/lucidrains/denoising-diffusion-pytorch
# %%
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision.utils as vutils
import torch
import tqdm
import cv2
from colorama import Fore
from colorama import Style


sample_steps = 20
data_path = './data/BRICKS_colorful_final'
model_dir = './results_colorful_final'
checkpoint_num = '100'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = sample_steps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False, # whether to calculate fid during training
    results_folder = model_dir       # folder to save results
)

trainer.load(checkpoint_num)


def show_multiple(imgs):
    npimg = vutils.make_grid(imgs, padding=2, nrow=10)
    npimg = npimg.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

def show_one(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)),
                            interpolation='nearest')
    plt.show()

# Is a copy of the sampling function from GaussianDiffusion but with noise as an input
def ddim_sample(model, noise, return_all_timesteps = False):
    shape = noise.shape
    batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], model.device, model.num_timesteps, model.sampling_timesteps, model.ddim_sampling_eta, model.objective

    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = noise 
    imgs = [img]

    x_start = None

    for time, time_next in tqdm.tqdm(time_pairs, desc = 'sampling loop time step'):
        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        self_cond = x_start if model.self_condition else None
        pred_noise, x_start, *_ = model.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

        if time_next < 0:
            img = x_start
            imgs.append(img)
            continue

        alpha = model.alphas_cumprod[time]
        alpha_next = model.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = model.unnormalize(ret)
    del imgs
    torch.cuda.empty_cache()
    return ret


# %%==========================================================================-
print(f"{Fore.GREEN}Here is one randomly generated sample")
sampled_images = diffusion.sample(batch_size = 1, return_all_timesteps=False)
show_one(sampled_images[0].detach().cpu())


# %%==========================================================================-
print(f"{Fore.GREEN}Here is the interpolation between these two randomly generated samples below")
torch.manual_seed(0)
noise1 = torch.randn((1, 3, 64, 64), device=device)
noise2 = torch.randn((1, 3, 64, 64)).to(device)

sampled_images = torch.stack([ddim_sample(diffusion, noise1, return_all_timesteps=False).detach().cpu()[0],
                              ddim_sample(diffusion, noise2, return_all_timesteps=False).detach().cpu()[0]])
show_multiple(sampled_images.detach().cpu())

img_list = []
steps = 10
for i in range(steps):
    interp_noise = torch.lerp(noise1, noise2, i/(steps-1))
    sampled_images = ddim_sample(diffusion, interp_noise, return_all_timesteps=False)
    img_list.append(sampled_images[0].detach().cpu())
del interp_noise
torch.cuda.empty_cache()
sampled_images = torch.stack(img_list).detach().cpu()
del img_list
torch.cuda.empty_cache()
show_multiple(sampled_images)


# %%==========================================================================-
def make_noise_images(seed, noise_vectors=20):
  torch.manual_seed(seed)
  noise = torch.randn(noise_vectors, 3, 64, 64, device=device)
  return noise

def interpolate_between_each_noise(noise, steps):
    img_list = []

    for j in range(len(noise)-1):
        for i in range(steps):
            interp_noise = torch.lerp(noise[j].unsqueeze(0), noise[j+1].unsqueeze(0), i/(steps-1))
            sampled_images = ddim_sample(diffusion, interp_noise, return_all_timesteps=False)
            img_list.append(sampled_images[0].detach().cpu())
    imgs = torch.stack(img_list)
    return imgs

def get_video_from_interpolations_forall_images(filename, frame_size=(64, 64), fps=30, noise_vectors=20, steps=20):
  codec = cv2.VideoWriter_fourcc(*'mp4v')
  video_writer = cv2.VideoWriter(filename, codec, fps, frame_size)

  noise = make_noise_images(1, noise_vectors)
  fake = interpolate_between_each_noise(noise, steps)

  for f in fake:
    img = vutils.make_grid(f)
    img = np.transpose(img, (1,2,0)).detach().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.array(img) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    video_writer.write(img)

  video_writer.release()

get_video_from_interpolations_forall_images('aaaa.mp4', fps=12, noise_vectors=10, steps=20)
# %%
