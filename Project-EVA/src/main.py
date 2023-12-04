# %%
from eva_modules.read_prompt import *
from eva_modules.speech import *
from eva_modules.tortoise_speech import *
from eva_modules.video import *
import time

import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip, afx, concatenate_audioclips, CompositeVideoClip
import soundfile as sf
import numpy as np
import yaml

import subprocess
import os
from dotenv import load_dotenv

load_dotenv()


LLM_PIPELINE = 'llm_pipeline.py'

def pipeline(
    img_prompts:List[str],
    tts_captions:List[str],
    speakers:List[str] = None,
    seed:int=69,
    device:str = "cuda",
    music_path = None
):
    '''
    tts_captions -> [VITS] -> wavs
    img_prompts -> [SD] -> imgs -> [SVD] -> videos
    wavs + videos -> final video
    '''
    if speakers is None:
        speakers = ["default"] * len(img_prompts)
    if not len(img_prompts) == len(tts_captions) == len(speakers):
        raise ValueError("Number of image prompts, number of text prompts, and number of speakers must match.")


    # define vars
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sampling_rate = config['video_generator']['audio_sampling_rate']
    fps = config['video_generator']['fps'] # TODO get this from type of SVD model
    vits_model = config['models']['vits']
    sd_model = config['models']['sd']
    svd_model = config['models']['svd']

    # get TTS model
    net_g, hps = get_tts_model(f"./configs/vits/{vits_model}.json", f"./models/vits/{vits_model}.pth") 

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    # generate all the audios first to know how long each videos should be
    generations = []
    for i, text in enumerate(tts_captions):
        # generate audio from text
        wav = text2speech(text, hps, net_g)
        sf.write(f'audio{i}.wav', wav, sampling_rate) 

        # generate sufficently long video from audio
        total_audio_seconds = get_seconds_in_wav(wav, hps.data.sampling_rate)
        num_generations = np.ceil(total_audio_seconds * fps / 25).astype(int)
        generations.append(num_generations)
    del net_g, hps

    # make all the audios the speaker
    for i, speaker in enumerate(speakers):
        if speaker != "default":
            if speaker.lower() not in map(lambda x: x.lower(), get_models()["so-vits-svc"]):
                print(f"Speaker {speaker} not found. Using default TTS. Use --listmodels to see available models.")
                continue
            speech2speech(model=speaker.lower(), input_path = f"audio{i}.wav", output_path = f"audio{i}.wav")

    # get clips
    videos = txt2vid(img_prompts, generations, sd_model, svd_model, seed=seed, device=device)
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

    # add music if there is
    if music_path is not None:
        music = AudioFileClip(music_path)
        if music.duration > final_clip.duration:
            print("Music is longer than the video, so we will trim it to match the video's length")
            music = music.subclip(0, final_clip.duration)
        elif music.duration < final_clip.duration:
            print("Music is shorter than the video, so we will loop it to match the video's length")
            music = afx.audio_loop(music, duration=final_clip.duration)
        audio = CompositeAudioClip([final_clip.audio.volumex(0.8), music.volumex(0.1)])
        final_clip = final_clip.set_audio(audio)

    final_clip.write_videofile("output.mp4", codec="libx264")
    return final_clip


def pipeline_tortoise(
    img_prompts:List[str],
    tts_captions:List[str],
    speakers:List[str] = None,
    seed:int=69,
    device:str = "cuda",
    music_path = None
):
    '''
    tts_captions -> [tortoise] -> torch wav
    img_prompts -> [SD] -> imgs -> [SVD] -> videos
    wavs + videos -> final video
    '''
    if speakers is None:
        speakers = ["default"] * len(img_prompts)
    if not len(img_prompts) == len(tts_captions) == len(speakers):
        raise ValueError("Number of image prompts, number of text prompts, and number of speakers must match.")


    # define vars
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sampling_rate = config['tortoise_tts']['audio_sampling_rate_tortoise']
    fps = config['video_generator']['fps']
    sd_model = config['models']['sd']
    svd_model = config['models']['svd']
    # get TTS model
    # if there is an error it is generally due to deepspeed, turn it to False if that happens to disable it
    tts = get_tts_model_tortoise(use_deepspeed=True, kv_cache=True, half=True) 

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    # create a character listing
    character_listing = create_voice_listing_characters(speakers, config['tortoise_tts'])

    # generate all the audios first to know how long each videos should be
    generations = []
    for i, (text, character) in enumerate(zip(tts_captions, characters)):
        # get voice from character
        voice = character_listing[character]

        # generate audio from text
        gen = text2speech_tortoise(text, tts, voice)
        save_audio_as_file_tortoise(f'audio{i}.wav', gen, sampling_rate)

        # generate sufficently long video from audio
        total_audio_seconds = get_seconds_in_wav(gen[0][0], sampling_rate)
        num_generations = np.ceil(total_audio_seconds * fps / 25).astype(int)
        generations.append(num_generations)
        del gen

    del tts

    # make all the audios the speaker
    for i, speaker in enumerate(speakers):
        if speaker != "default":
            if speaker.lower() not in map(lambda x: x.lower(), get_models()["so-vits-svc"]):
                print(f"Speaker {speaker} not found. Using default TTS. Use --listmodels to see available models.")
                continue
            speech2speech(model=speaker.lower(), input_path = f"audio{i}.wav", output_path = f"audio{i}.wav")

    # get clips
    videos = txt2vid(img_prompts, generations, sd_model, svd_model, seed=seed, device=device)
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

    # add music if there is
    if music_path is not None:
        music = AudioFileClip(music_path)
        if music.duration > final_clip.duration:
            print("Music is longer than the video, so we will trim it to match the video's length")
            music = music.subclip(0, final_clip.duration)
        elif music.duration < final_clip.duration:
            print("Music is shorter than the video, so we will loop it to match the video's length")
            music = afx.audio_loop(music, duration=final_clip.duration)
        audio = CompositeAudioClip([final_clip.audio.volumex(0.8), music.volumex(0.1)])
        final_clip = final_clip.set_audio(audio)

    final_clip.write_videofile("output.mp4", codec="libx264")
    return final_clip

def get_models(print_models=False):
    '''
    returns a dict of models
    TODO add checking for sd and svd model is in this list
    '''
    models_path = Path("./models/")
    all_models = {}
    for model in models_path.iterdir():
        if model.is_dir():
            all_models[model.stem] = [checkpoint.stem for checkpoint in model.iterdir()]

    if print_models:
        print("Available models:")
        for model in all_models:
            print(f"{model}:")
            for checkpoint in all_models[model]:
                print(f"  - {checkpoint}")
    return all_models


def get_prompt(input):
    arguments = ["python3", f"{LLM_PIPELINE}", f"'{input}'"]
    subprocess.run(arguments, capture_output=False, text=True)
    llm_data_reply = os.environ['LLM_DATA_REPLY']
    return llm_data_reply


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--selfprompt', 
                        type=str, 
                        help="PATH to the a prompt that is created by the user (must follow the similar file format), which is the path to a .json file. If not prompts are used, then the test_prompt.json will be defaulted.",
                        default="test_prompt.json",
                        required=False)
    
    parser.add_argument('--prompt', 
                        type=str, 
                        help="a prompt to queue a LLM to generate a movie",
                        required=False) 

    parser.add_argument('--seed',
                        type=int,
                        help="Seed for the random number generators",
                        default=69)
    parser.add_argument('--music',
                        type=str,
                        help="PATH to the music, if we want to add to the final track",
                        required=False)
    parser.add_argument('--listmodels',
                        action='store_true',
                        help="List the available models and exit",
                        default=False)
    args = parser.parse_args()

    prompt = args.selfprompt

    # use the LLM
    if args.prompt is not None:
        if args.listmodels:
            get_models(print_models=True)
            exit()
        
        prompt = get_prompt(args.prompt)

    print(f'using the prompt at: {prompt}')

    captions, dialogues, characters = read_prompt(prompt)
    print(captions)

    s = time.perf_counter()
    pipeline_tortoise(captions, dialogues, characters, music_path=args.music, seed=args.seed, device=device)
    e = time.perf_counter()
    print(f"Time elapsed: {e-s} seconds")

# %%
