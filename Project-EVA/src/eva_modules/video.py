from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from typing import Optional

def txt2img(prompt:str):
    '''
    SD
    returns img
    '''
    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]  
    # del pipe, scheduler
    return image

def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    num_continuous_samples: int = 1,
):
    '''
    TODO change params and implement it here

    SVD from stability ai
    return: numpy/tensors of frames (num_frames, height, width, 3)
    '''
    pass

def txt2vid():
    '''
    SD
    returns video
    '''
    # img = txt2img()
    # sample()
    pass