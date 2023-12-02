# %%
from eva_modules.read_prompt import *
from eva_modules.speech import *
from eva_modules.video import *
import time
import torch

import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip, concatenate_audioclips, CompositeVideoClip
import soundfile as sf
import numpy as np
import yaml

from torch.utils.data import DataLoader
import sys

def pipeline(
    img_prompts:List[str],
    tts_captions:List[str],
    speakers:List[str] = None,
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
):
    '''
    TODO implement svc for speakers

    tts_captions -> [VITS] -> wavs
    img_prompts -> [SD] -> imgs -> [SVD] -> videos
    wavs + videos -> final video
    '''
    if speakers is None:
        speakers = ["default"] * len(img_prompts)
    if not len(img_prompts) == len(tts_captions) == len(speakers):
        raise ValueError("Number of image prompts, number of text prompts, and number of speakers must match.")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get TTS
    # TODO change this for speaker selection
    net_g, hps = get_tts("./vits/configs/vctk_base.json", "./vits/models/pretrained_vctk.pth") 

    # define vars
    # TODO get this from type of SVD model
    sampling_rate = config['video_generator']['audio_sampling_rate']
    fps = config['video_generator']['fps']

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    # generate all the audios first to know how long each videos should be
    generations = []
    for i, text in enumerate(tts_captions):
        # generate audio from text
        wav = tts(text, hps, net_g)
        sf.write(f'audio{i}.wav', wav, sampling_rate) 

        # generate sufficently long video from audio
        total_audio_seconds = get_seconds_in_wav(wav, hps.data.sampling_rate)
        num_generations = np.ceil(total_audio_seconds * fps / 25).astype(int)
        generations.append(num_generations)
    del net_g, hps

    # get clips
    videos = txt2vid(img_prompts, generations, device=device)
    clips = []
    for i, video in enumerate(videos):
        # transform the video to match the sequence of images required for an image sequence clip
        transform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).cpu().numpy()]
        clip = ImageSequenceClip(transform_video, fps=fps) 

        # add audio to clip and fix the duration and glitchiness
        audio = AudioFileClip(f'audio{i}.wav')
        audio = audio.subclip(0, audio.duration-0.05)
        silence_duration = max(0, clip.duration)
        audio_silence = AudioClip(lambda x: 0, duration=silence_duration, fps=sampling_rate)
        audio = CompositeAudioClip([audio, audio_silence])

        clip = clip.set_audio(audio)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("output.mp4", codec="libx264")
    return final_clip

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', 
                        type=str, 
                        help="PATH to the prompt, which is the path to a .json file",
                        required=True)

    args = parser.parse_args()

    read_prompt(args.prompt)


    # svc()

    # prompts = ["A sheep with a gold chain around its neck is standing in a field."]
    # images = txt2img(prompts, device=device)
    # sample(images, version="svd_xt")

    # time.sleep(10)
    # txt2img()

    # text = "Hello world! My name is Zhi Zheng."
    # text = "My name is Walter Hartwell White"
    # text = "The significance of this paper lies in its efforts to mitigate a fundamental flaw in deep learning models: their vulnerability to adversarial attacks."
    # texts = [
    #     "Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng.",
    #     "Through the rise of image processing through machine learning, adversarial attacks prove to be a threat to the robustness of models.",
    #     "This paper examines a novel purification technique that leverages the robustness of diffusion and model attention.",
    #     "Hi my name is Zhi Zheng.",
    #     "We apply inpainting to the attention map of a high layer representation of the model in order to effectively target an area to diffuse.",
    #     "This provides a more effective way to purify an attacked image by generalizing the dimensions of which the model classifies.",
    #     "Through experimentation we provide applicable methods that effectively eliminate adversarial attacks."
    # ] * a
    # testing()

    captions, dialogues, characters = parse_json(args.prompt)

    # samples = txt2vid("", num_images=2, device=device)
    pipeline(captions, dialogues, device=device)
