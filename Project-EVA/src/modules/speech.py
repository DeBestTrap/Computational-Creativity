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
    from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip
    from PIL import Image
    import soundfile as sf
    import numpy as np

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    # for s in TTS().list_models():
    #     print(s)

    tts = TTS('tts_models/en/ljspeech/vits').to(device)
    # tts = TTS('./pretrained_vctk.pth').to(device)
    # tts = TTS('tts_models/en/ljspeech/vits--neon').to(device)

    # text = "Hello world! My name is Zhi Zheng."
    # text = "My name is Walter Hartwell White"
    text = "The significance of this paper lies in its efforts to mitigate a fundamental flaw in deep learning models: their vulnerability to adversarial attacks."
    wav = tts.tts(text=text)
    print(type(wav))
    # tts.tts_to_file(text=text, file_path="output.wav")

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        # convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    # define vars
    sampling_rate = 22050 # sampling rate of the audio
    fps = 6 # frames per second of the video
    seconds_per_video = 25 / fps

    total_audio_seconds = get_seconds_in_wav(wav)

    video = torch.rand(25, 3, 512, 512)
    seconds_overall = seconds_per_video

    while seconds_overall < total_audio_seconds:
        # generate the video
        random_video = torch.rand(25, 3, 512, 512)

        video = torch.cat((video, random_video), dim=0)

        seconds_overall += seconds_per_video

    print(video.shape)

    # pil_images = [Image.fromarray((np_img * 255).astype('uint8')) for np_img in video.permute(0, 2, 3, 1).numpy()]
    pil_images = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).numpy()]

    # get audio
    wav_numpy = np.array(wav)
    sf.write('audio.wav', wav_numpy, sampling_rate) 

    # get clip
    clip = ImageSequenceClip(pil_images, fps=fps) 

    # add audio to clip
    audio = AudioFileClip('audio.wav')
    composite_audio = CompositeAudioClip([audio.set_duration(audio.duration)])
    clip = clip.set_audio(composite_audio)

    # write out
    clip.write_videofile("output.mp4", codec="libx264")