# Based on https://github.com/lucidrains/denoising-diffusion-pytorch
# %%
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision.utils as vutils
import torch
import tqdm
import cv2
import yaml
import os
from colorama import Fore
from colorama import Style

sample_steps = 50
data_path = './data/BRICKS_colorful_final'
results_path = './results_colorful_final'
checkpoint_num = '100'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(os.path.join(results_path, 'config.yaml'), "r") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)
    print(parameters)
# %%

model = Unet(
    dim = parameters['unet']['dim'],
    dim_mults = parameters['unet']['dim_mults'],
    flash_attn = parameters['unet']['flash_attn'] 
)

diffusion = GaussianDiffusion(
    model,
    image_size = parameters['diffusion']['image_size'],
    timesteps = parameters['diffusion']['timesteps'],           # number of steps
    sampling_timesteps = sample_steps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size = parameters['trainer']['train_batch_size'],
    train_lr = parameters['trainer']['train_lr'],
    train_num_steps = parameters['trainer']['train_num_steps'],         # total training steps
    gradient_accumulate_every = parameters['trainer']['gradient_accumulate_every'],    # gradient accumulation steps
    ema_decay = parameters['trainer']['ema_decay'],                # exponential moving average decay
    amp = parameters['trainer']['amp'],                       # turn on mixed precision
    calculate_fid = parameters['trainer']['calculate_fid'], # whether to calculate fid during training
    results_folder = results_path       # folder to save results
)


if checkpoint_num:
    trainer.load(checkpoint_num)


def show_multiple(imgs, nrows=10):
    npimg = vutils.make_grid(imgs, padding=2, nrow=nrows)
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

def generate_img(noise):
    if noise.shape == (3, 64, 64):
        noise = noise.unsqueeze(0)
    elif noise.shape != (1, 3, 64, 64):
        raise ValueError(f"noise must be in the shape of (1, 3, 64, 64) or (3, 64, 64)\nnoise is shape: {noise.shape}")
    n = noise.to(device)
    img = ddim_sample(diffusion, n, return_all_timesteps=False).detach().cpu()
    del n
    torch.cuda.empty_cache()
    return img

def generate_imgs(noises):
    imgs = []
    for n in noises:
        imgs.append(generate_img(n))
    return torch.vstack(imgs)



# %%==========================================================================-
print(f"{Fore.GREEN}Here is a couple randomly generated samples")
imgs = []
for _ in range(9):
    sampled_images = diffusion.sample(batch_size = 1, return_all_timesteps=False)
    imgs.append(sampled_images[0].detach().cpu())
show_multiple(imgs, nrows=3)
del imgs
torch.cuda.empty_cache()

# %%==========================================================================-
print(f"{Fore.GREEN}Here is the interpolation between these two randomly generated samples below")
torch.manual_seed(0)
noise1 = torch.randn((1, 3, 64, 64))
noise2 = torch.randn((1, 3, 64, 64))

sampled_images = generate_imgs([noise1, noise2])
show_multiple(sampled_images)

steps = 10
sampled_images = generate_imgs([torch.lerp(noise1, noise2, i/(steps-1)) for i in range(steps)])
show_multiple(sampled_images)

# %%==========================================================================-
print(f"{Fore.GREEN}Here is the interpolation between 4 randomly generated samples")

torch.manual_seed(1337)

steps = 10
noise1, noise2, noise3, noise4 = torch.randn((4, 3, 64, 64), device="cpu")

n1_2 = torch.stack([torch.lerp(noise1, noise2, i/(steps-1)) for i in range(steps)])
n3_4 = torch.stack([torch.lerp(noise3, noise4, i/(steps-1)) for i in range(steps)])

n1_2_3_4 = []
for j in range(n1_2.shape[0]):
    interp_step = torch.stack([torch.lerp(n1_2[j], n3_4[j], i/(steps-1)) for i in range(steps)])
    n1_2_3_4.append(interp_step)
n1_2_3_4 = torch.vstack(n1_2_3_4)

sampled_images = generate_imgs(n1_2_3_4)
show_multiple(sampled_images, nrows=steps)

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

get_video_from_interpolations_forall_images('diffusion_interpolation.mp4', fps=30, noise_vectors=10, steps=100)
# %%
