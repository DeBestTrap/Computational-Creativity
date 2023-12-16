from typing import List
import yaml
from pathlib import Path
from t2m_modules.video import *
from t2m_modules.speech import *
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip, afx
from colorama import Fore, Style, Back

class Pipeline:
    def __init__(self,
                 use_tortoise:bool=False,
                 width:int=1204,
                 height:int=576,
                 seed:int=69,
                 device:str = "cuda",
                 output_dir:Path = None
                 ) -> None:
                 
        self.use_tortoise = use_tortoise
        self.width = width
        self.height = height
        self.seed = seed
        self.device = device
        self.output_dir = output_dir
    

    def run(self,
        img_prompts:List[str],
        tts_captions:List[str],
        speakers:List[str] = None,
        style:str = "realistic",
        music_path = None,
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
        config_tts = config["tortoise_tts"] if self.use_tortoise else config["vits"] 
        sampling_rate = config_tts['audio_sampling_rate']
        fps = config['video_generator']['fps']
        sd_model = config['models']['sd']
        svd_model = config['models']['svd']

        # get TTS model
        # if there is an error it is generally due to deepspeed, turn it to False if that happens to disable it
        # tts = get_tts_model_tortoise(use_deepspeed=False, kv_cache=True, half=True) 
        if self.use_tortoise:
            pipe_tts = Text2Speech(use_tortoise=True,
                                   tortoise_use_deepspeed=False,
                                   tortoise_kv_cache=True,
                                   tortoise_half=True
                                   )
        else:
            pipe_tts = Text2Speech(use_tortoise=False,
                                vits_config_path=Path("configs/vits/"),
                                vits_model_path=Path("models/vits/"),
                                vits_list_configs=get_configs()["vits"],
                                vits_list_models=get_models()["vits"],
                                )

        # create a character listing
        character_listing = create_voice_listing_characters(speakers, config_tts)

        # generate all the audios first to know how long each videos should be
        print(f"{Fore.GREEN}-= Generating audio =-{Style.RESET_ALL}")
        generations = []
        for i, (text, character) in enumerate(zip(tts_captions, speakers)):
            # get voice from character
            voice = character_listing[character]

            # generate audio from text
            wav = pipe_tts.text2speech(text, voice)
            audio_path = self.output_dir / f'dialogue/{i}.wav'
            pipe_tts.save_audio_as_file(audio_path, wav, sampling_rate)

            # generate sufficently long video from audio
            total_audio_seconds = self.get_seconds_in_wav(wav[0], sampling_rate)
            num_generations = np.ceil(total_audio_seconds * fps / 25).astype(int)
            generations.append(num_generations)
        del pipe_tts

        # make all the audios the speaker
        # TODO maybe make this selectable by user
        print(f"{Fore.GREEN}-= Voice Conversion =-{Style.RESET_ALL}")
        os.makedirs(self.output_dir / "dialogue_converted", exist_ok=True)
        for i, speaker in enumerate(speakers):
            if speaker != "default":
                if speaker.lower() not in map(lambda x: x.lower(), get_models()["so-vits-svc"]):
                    print(f"{Fore.RED}Speaker {speaker} not found. Using default TTS. Use --listmodels to see available models.{Style.RESET_ALL}")
                else:
                    print(f"Converting from {character_listing[speaker]} to {speaker}")
                    speech2speech(model=speaker.lower(), input_path = self.output_dir/"dialogue"/f"{i}.wav", output_path = self.output_dir/"dialogue_converted"/f"{i}.wav")

        # get clips
        print(f"{Fore.GREEN}-= Generating video =-{Style.RESET_ALL}")
        pipe_t2v = Text2Video(model_dir=Path("models/"),
                              sd_model=sd_model,
                              is_sdxl=True,
                              svd_model=svd_model,
                              seed=self.seed,
                              device=self.device,
                              output_dir=self.output_dir,
                              width=self.width,
                              height=self.height
                              )
        videos = pipe_t2v.text2vid(img_prompts, style, generations)
        del pipe_t2v
        clips = []
        for i, video in tqdm(enumerate(videos), desc="Combining videos and audio", total=len(videos)):
            # transform the video to match the sequence of images required for an image sequence clip
            transform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).cpu().numpy()]
            clip = ImageSequenceClip(transform_video, fps=fps) 

            # fix the duration and glitchiness of audio and add to clip
            converted_audio_path = self.output_dir/'dialogue_converted'/f'{i}.wav'
            audio_path = self.output_dir/'dialogue'/f'{i}.wav' if not os.path.exists(converted_audio_path) else converted_audio_path
            audio = AudioFileClip(str(audio_path)).fx(afx.audio_normalize)
            audio = audio.subclip(0, audio.duration-0.05)
            silence_duration = max(0, clip.duration)
            audio_silence = AudioClip(lambda x: 0, duration=silence_duration, fps=sampling_rate)
            audio = CompositeAudioClip([audio, audio_silence])
            clip = clip.set_audio(audio)
            clips.append(clip)

        final_clip = concatenate_videoclips(clips)

        # add music if there is
        if music_path is not None:
            music = AudioFileClip(music_path).fx(afx.audio_normalize)
            if music.duration > final_clip.duration:
                print("Music is longer than the video, so we will trim it to match the video's length")
                music = music.subclip(0, final_clip.duration)
            elif music.duration < final_clip.duration:
                print("Music is shorter than the video, so we will loop it to match the video's length")
                music = afx.audio_loop(music, duration=final_clip.duration)
            audio = CompositeAudioClip([final_clip.audio.volumex(0.8), music.volumex(0.03)])
            final_clip = final_clip.set_audio(audio)

        output_path = self.output_dir / Path("movie.mp4")
        final_clip.write_videofile(str(output_path), codec="libx264")
        return final_clip

    def get_seconds_in_wav(self,
                           wav:torch.Tensor,
                           sampling_rate:int):
        return len(wav) / sampling_rate


def get_models(print_models=False):
    '''
    returns a dict of models
    TODO add checking for sd and svd model is in this list
    '''
    models_path = Path("./models/")
    all_models = {}
    for model in models_path.iterdir():
        if model.is_dir():
            all_models[model.stem] = [checkpoint.stem for checkpoint in model.iterdir() if checkpoint.suffix != ".txt"]

    if print_models:
        print("Available models:")
        for model in all_models:
            print(f"{model}:")
            for checkpoint in all_models[model]:
                print(f"  - {checkpoint}")
    return all_models

def get_configs(print_configs=False):
    '''
    returns a dict of configs
    '''
    configs_path = Path("./configs/")
    all_configs = {}
    for config in configs_path.iterdir():
        if config.is_dir():
            all_configs[config.stem] = [config_file.stem for config_file in config.iterdir() if config_file.suffix != ".txt"]

    if print_configs:
        print("Available configs:")
        for config in all_configs:
            print(f"{config}:")
            for config_file in all_configs[config]:
                print(f"  - {config_file}")
    return all_configs