from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from typing import Optional, List
from PIL import Image

def txt2img(prompts:List[str], device:str):
    '''
    SD
    returns img
    '''
    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    images = []
    for prompt in prompts:
        image = pipe(prompt).images[0]  
        images.append(image)
    # del pipe, scheduler
    return images

# def sample(
#     input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
#     num_frames: Optional[int] = None,
#     num_steps: Optional[int] = None,
#     version: str = "svd",
#     fps_id: int = 6,
#     motion_bucket_id: int = 127,
#     cond_aug: float = 0.02,
#     seed: int = 23,
#     decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
#     device: str = "cuda",
#     output_folder: Optional[str] = None,
#     num_continuous_samples: int = 1,
# ):
#     '''
#     TODO change params and implement it here

#     SVD from stability ai
#     return: numpy/tensors of frames (num_frames, height, width, 3)
#     '''
#     pass

import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
import PIL.Image
from torchvision.transforms import ToTensor

# from scripts.util.detection.nsfw_and_watermark_dectection import \
#     DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config


def sample(
    # input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    input_imgs: List[PIL.Image.Image],
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    # num_continuous_samples: List[int] = None,
    num_continuous_samples: int = 1,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    print(f"decoding_t: {decoding_t}")
    torch.cuda.set_device(1)
    # if num_continuous_samples is None:
    #     num_continuous_samples = [1] * len(input_imgs)

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        "generative-models/"+model_config,
        device,
        num_frames,
        num_steps,
    )
    torch.manual_seed(seed)
    videos = []

    for i, image in enumerate(input_imgs):
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = ToTensor()(image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device) 
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        samples = torch.empty((0,) + image.shape[1:], device=device)

        for _ in range(num_continuous_samples):
        # for _ in range(num_continuous_samples[i]):
            with torch.no_grad():
                with torch.autocast(device):
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=[
                            "cond_frames",
                            "cond_frames_without_noise",
                        ],
                    )

                    for k in ["crossattn", "concat"]:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                    randn = torch.randn(shape, device=device)

                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(
                        2, num_frames
                    ).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c):
                        return model.denoiser(
                            model.model, input, sigma, c, **additional_model_inputs
                        )

                    samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                    model.en_and_decode_n_samples_a_time = decoding_t
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.concat((samples, torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)), dim=0)

                    image = samples_x[-1][None, :, :, :]
                    value_dict["cond_frames_without_noise"] = image
                    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

        # os.makedirs(output_folder, exist_ok=True)
        # base_count = len(glob(os.path.join(output_folder, "*.mp4")))
        # video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
        # writer = cv2.VideoWriter(
        #     video_path,
        #     cv2.VideoWriter_fourcc(*"MP4V"),
        #     fps_id + 1,
        #     (samples.shape[-1], samples.shape[-2]),
        # )

        samples = embed_watermark(samples)
        videos.append(samples)
        # samples = filter(samples)
        # vid = (
        #     (rearrange(samples, "t c h w -> t h w c") * 255)
        #     .cpu()
        #     .numpy()
        #     .astype(np.uint8)
        # )
        # for frame in vid:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     writer.write(frame)
        # writer.release()
        return samples

    return videos


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    # if device == "cuda":
    if "cuda" in device:
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    # if device == "cuda":
    if "cuda" in device:
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    filter = lambda x: x
    return model, filter

def txt2vid(prompts, num_images=1, generations=None, device="cuda"):
    '''
    SD
    returns video
    '''
    # if generations is None:
    #     generations = [1]*num_images

    prompts = ["A sheep with a gold chain around its neck is standing in a field."]*num_images
    images = txt2img(prompts, device=device)
    # for i in range(len(images)):
    #     images[i] = images[i].resize((1024, 576))
    # samples = sample(images, version="svd_xt", num_continuous_samples=[generations]*num_images)
    samples = sample(images, version="svd_xt", num_continuous_samples=generations)
    return samples