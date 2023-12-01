import torch
from TTS.api import TTS
from typing import List

def txt2speech(text:List[str], device:str):
    '''
    returns ??
    '''
    pass
    tts = TTS('tts_models/en/ljspeech/vits').to(device)
    # tts = TTS('./pretrained_vctk.pth').to(device)
    # tts = TTS('tts_models/en/ljspeech/vits--neon').to(device)

    wav = tts.tts(text=text)
    print(type(wav))
    return wav

def speech2speech(model):
    '''
    makes waveform sound like the model's voice
    returns ??
    '''
    pass

def testing():
    import torch
    from TTS.api import TTS
    from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
    import soundfile as sf
    import numpy as np
    import yaml

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get TTS
    tts = TTS('tts_models/en/ljspeech/vits').to(device)

    # define vars
    sampling_rate = config['video_generator']['audio_sampling_rate']
    fps = config['video_generator']['fps']
    seconds_per_video = 25 / fps

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        # convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    def generate_video(text):
        video = torch.rand(25, 3, 512, 512)
        return video

    texts = [
        "Through the rise of image processing through machine learning, adversarial attacks prove to be a threat to the robustness of models.",
        "This paper examines a novel purification technique that leverages the robustness of diffusion and model attention.",
        "We apply inpainting to the attention map of a high layer representation of the model in order to effectively target an area to diffuse.",
        "This provides a more effective way to purify an attacked image by generalizing the dimensions of which the model classifies.",
        "Through experimentation we provide applicable methods that effectively eliminate adversarial attacks."
    ]

    # go through every iteration of text
    clips = []
    for text in texts:
        # generate initial video
        video = generate_video(text)
        wav = tts.tts(text=text)
    
        total_audio_seconds = get_seconds_in_wav(wav)

        seconds_overall = seconds_per_video
        while seconds_overall < total_audio_seconds:
            # generate the video
            random_video = generate_video(text)

            video = torch.cat((video, random_video), dim=0)

            seconds_overall += seconds_per_video

        # get audio
        wav_numpy = np.array(wav)
        sf.write('audio.wav', wav_numpy, sampling_rate) 

        # get clip
        # transform the video to match the sequence of images required for an image sequence clip
        tranform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).numpy()]
        clip = ImageSequenceClip(tranform_video, fps=fps) 

        # add audio to clip
        audio = AudioFileClip('audio.wav')
        composite_audio = CompositeAudioClip([audio.set_duration(audio.duration)])
        clip = clip.set_audio(composite_audio)

        # add to all clips
        clips.append(clip)


    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("output.mp4", codec="libx264")