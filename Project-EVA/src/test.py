# %%
from eva_modules.read_prompt import *
# from eva_modules.speech import *
# from eva_modules.video import *
import time

import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip, concatenate_audioclips, CompositeVideoClip
import soundfile as sf
import numpy as np
import yaml
from typing import List

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

import sys
sys.path.insert(0, './tortoise-tts')
import tortoise.api_fast as api
from tortoise import utils
from tortoise.utils.audio import load_audio, load_voice, load_voices

def pipeline(
    img_prompts:List[str],
    tts_captions:List[str],
    speakers:List[str] = None,
    seed:int=69,
    device:str = "cuda" if torch.cuda.is_available() else "cpu",
    music_path = None
):
    '''
    tts_captions -> [speecht5] -> wavs
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
    tts = api.TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)

    # Pick one of the voices from the output above
    voice = 'tom'
    # Load it and send it through Tortoise.
    voice_samples, conditioning_latents = load_voice(voice)
    preset = "ultra_fast"

    # define vars
    sampling_rate = config['video_generator']['audio_sampling_rate_speecht5']
    fps = config['video_generator']['fps']

    def get_seconds_in_wav(wav, sampling_rate):
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    # generate all the audios first to know how long each videos should be
    generations = []
    for i, text in enumerate(tts_captions):
        # generate audio from text
        wav = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)

        sf.write(f'audio{i}.wav', wav, sampling_rate)

        # generate sufficently long video from audio
        total_audio_seconds = get_seconds_in_wav(wav, sampling_rate)
        num_generations = np.ceil(total_audio_seconds * fps / 25).astype(int)
        generations.append(num_generations)
    # del net_g, hps

    # get clips
    # videos = txt2vid(img_prompts, generations, device=device)
    # clips = []
    # for i, video in enumerate(videos):
    #     # transform the video to match the sequence of images required for an image sequence clip
    #     transform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).cpu().numpy()]
    #     clip = ImageSequenceClip(transform_video, fps=fps) 

    #     # add audio to clip and fix the duration and glitchiness
    #     audio = AudioFileClip(f'audio{i}.wav')
    #     audio = audio.subclip(0, audio.duration-0.05)
    #     silence_duration = max(0, clip.duration)
    #     audio_silence = AudioClip(lambda x: 0, duration=silence_duration, fps=sampling_rate)
    #     audio = CompositeAudioClip([audio, audio_silence])

    #     clip = clip.set_audio(audio)
    #     clips.append(clip)

    # final_clip = concatenate_videoclips(clips)

    # # add music if there is
    # if music_path is not None:
    #     music_clip = AudioFileClip(music_path)
        
    #     audio = CompositeAudioClip([final_clip.audio.volumex(0.8), music_clip.volumex(0.1)])
    #     final_clip = final_clip.set_audio(audio)

    # final_clip.write_videofile("output.mp4", codec="libx264")
    # return final_clip

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', 
                        type=str, 
                        help="PATH to the prompt, which is the path to a .json file",
                        required=True)
    parser.add_argument('--seed',
                        type=int,
                        help="Seed for the random number generators",
                        default=69)
    parser.add_argument('--music',
                        type=str,
                        help="PATH to the music, if we want to add to the final track",
                        required=False)

    args = parser.parse_args()

    captions, dialogues, characters = read_prompt(args.prompt)

    s = time.perf_counter()
    pipeline(captions, dialogues, seed=args.seed, device=device)
    e = time.perf_counter()
    print(f"Time elapsed: {e-s} seconds")

# %%