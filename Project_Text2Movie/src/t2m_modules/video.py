from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from typing import Optional, List
from pathlib import Path

import math
from einops import rearrange, repeat
from omegaconf import OmegaConf
import PIL.Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

# from scripts.util.detection.nsfw_and_watermark_dectection import \
#     DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config

from moviepy.editor import ImageSequenceClip


class Text2Video:
    def __init__(self,
                 model_dir: Path,
                 sd_model: Optional[str],
                 svd_model: str,
                 is_sdxl: bool,
                 output_dir: Path,
                 width: int = 1024,
                 height: int = 576,
                 seed: int = 69,
                 device = "cuda"
                 ) -> None:

        if (model_dir/"sd"/sd_model) is None or (model_dir/"sdxl"/sd_model) is None:
            print(f"SD model {sd_model} does not exist. Using default model.")
            self.use_default_sd = True
            if is_sdxl:
                self.sd_model = "stabilityai/stable-diffusion-xl-base-1.0"
                self.is_sdxl = True
            else:
                self.sd_model = "stabilityai/stable-diffusion-2-1-base"
                self.is_sdxl = False
        else:
            self.use_default_sd = False
            self.sd_model = sd_model
            self.is_sdxl = is_sdxl


        if (model_dir/"svd"/svd_model) is None:
            raise ValueError(f"SVD model {svd_model} does not exist.")

        self.svd_model = svd_model
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.seed = seed
        self.width = width
        self.height = height
        self.device = device

    def text2vid(self,
                prompts: List[str],
                style: str,
                generations: List[int] = None,
    ):
        '''
        prompts -> [SD] -> images -> [SVD] -> videos
        returns video
        '''
        if generations is None:
            generations = [1]*len(prompts)

        images = self.txt2img(prompts, style)
        samples = self.img2vid(images, num_continuous_samples=generations)
        return samples


    def txt2img(self,
                prompts:List[str],
                style:str,
                loras:List[str] = None,
    
    ) -> List[PIL.Image.Image]:
        '''
        SD
        returns List[PIL.Image.Image]
        '''
        # torch.cuda.set_device(1)

        if self.is_sdxl:
            if self.use_default_sd:
                # SDXL normal
                pipe = StableDiffusionXLPipeline.from_pretrained(self.sd_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to(self.device)
                # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                #     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
                # ).to("cuda")
            else:
                # SDXL from file
                model_path = str(self.model_dir/"sdxl"/self.sd_model)
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to(self.device)
                # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            if self.use_default_sd:
                # SD normal
                scheduler = EulerDiscreteScheduler.from_pretrained(self.sd_model, subfolder="scheduler")
                pipe = StableDiffusionPipeline.from_pretrained(self.sd_model, scheduler=scheduler, torch_dtype=torch.float16
                ).to(self.device)
            else:
                # SD from file
                # self.sd_model = "models/sd/realismFromHades_v81HQ.safetensors"
                model_path = str(self.model_dir/"sd"/self.sd_model)
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path, torch_dtype=torch.float16)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config
                ).to(self.device)

        # generate images
        images = []
        for i, prompt in tqdm(enumerate(prompts), desc="Generating images", total=len(prompts)):
            prompt += f" in a {style} style"
            generator = torch.Generator(device="cuda").manual_seed(self.seed+i) 
            image = pipe(prompt, height=self.height, width=self.width, generator=generator).images[0]  
            images.append(image)
        
        # save images
        for i, image in enumerate(images):
            image.save(self.output_dir / f"images/{i}.png")

        return images

    def img2vid(
        self,
        input_imgs: List[PIL.Image.Image],
        num_frames: Optional[int] = None,
        num_steps: Optional[int] = None,
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        num_continuous_samples: List[int] = None,
    ) -> List[torch.Tensor]:
        """
        Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
        image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.

        SVD from stability ai
        return: list of tensors of frames of size (num_frames, height, width, 3)
        """

        # uncomment to use a second GPU for SVD
        # torch.cuda.set_device(1)

        if num_continuous_samples is None:
            num_continuous_samples = [1] * len(input_imgs)

        if self.svd_model == "svd":
            num_frames = default(num_frames, 14)
            num_steps = default(num_steps, 25)
            model_config = "configs/svd/svd.yaml"
        elif self.svd_model == "svd_xt":
            num_frames = default(num_frames, 25)
            num_steps = default(num_steps, 30)
            model_config = "configs/svd/svd_xt.yaml"
        elif self.svd_model == "svd_image_decoder":
            num_frames = default(num_frames, 14)
            num_steps = default(num_steps, 25)
            model_config = "configs/svd/svd_image_decoder.yaml"
        elif self.svd_model == "svd_xt_image_decoder":
            num_frames = default(num_frames, 25)
            num_steps = default(num_steps, 30)
            model_config = "configs/svd/svd_xt_image_decoder.yaml"
        else:
            raise ValueError(f"Version {self.svd_model} does not exist.")

        model, filter = load_model(
            # "generative-models/"+model_config,
            model_config,
            self.device,
            num_frames,
            num_steps,
        )

        torch.manual_seed(self.seed)
        videos = []
        for i, image in tqdm(enumerate(input_imgs), desc="Generating videos", total=len(input_imgs)):
            w, h = image.size

            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )

            image = ToTensor()(image)
            image = image * 2.0 - 1.0

            image = image.unsqueeze(0).to(self.device) 
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

            samples = torch.empty((0,) + image.shape[1:])
            for _ in range(num_continuous_samples[i]):
                with torch.no_grad():
                    with torch.autocast(self.device):
                        batch, batch_uc = get_batch(
                            get_unique_embedder_keys_from_conditioner(model.conditioner),
                            value_dict,
                            [1, num_frames],
                            T=num_frames,
                            device=self.device,
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

                        randn = torch.randn(shape, device=self.device)

                        additional_model_inputs = {}
                        additional_model_inputs["image_only_indicator"] = torch.zeros(
                            2, num_frames
                        ).to(self.device)
                        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                        def denoiser(input, sigma, c):
                            return model.denoiser(
                                model.model, input, sigma, c, **additional_model_inputs
                            )

                        samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                        model.en_and_decode_n_samples_a_time = decoding_t
                        samples_x = model.decode_first_stage(samples_z)
                        samples = torch.concat((samples, torch.clamp((samples_x.cpu() + 1.0) / 2.0, min=0.0, max=1.0)), dim=0)

                        image = samples_x[-1][None, :, :, :]
                        value_dict["cond_frames_without_noise"] = image
                        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
                        del samples_x, samples_z, randn, c, uc, batch, batch_uc, additional_model_inputs

            samples = embed_watermark(samples).cpu()
            videos.append(samples)
        
        # save videos
        for i, video in enumerate(videos):
            transform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).cpu().numpy()]
            clip = ImageSequenceClip(transform_video, fps=fps_id) 
            video_path = self.output_dir / f"videos/{i}.mp4"
            clip.write_videofile(str(video_path), codec="libx264")

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
    if "cuda" in device:
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if "cuda" in device:
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    # filter = DeepFloydDataFiltering(verbose=False, device=device)
    filter = lambda x: x
    return model, filter