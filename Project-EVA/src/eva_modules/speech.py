import torch
from typing import List

def txt2speech(text:List[str], device:str):
    '''
    returns ??
    '''


def speech2speech(model):
    '''
    makes waveform sound like the model's voice
    returns ??
    '''
    pass


import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, AudioClip, concatenate_audioclips, CompositeVideoClip
import soundfile as sf
import numpy as np
import yaml

from torch.utils.data import DataLoader
import sys

sys.path.insert(0, './vits')
import commons as commons
import utils as utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# import vits.commons as commons
# import vits.utils as utils
# from vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
# from vits.models import SynthesizerTrn
# from vits.text.symbols import symbols
# from vits.text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def tts(text, hps, net_g):
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    return audio

def get_tts_hps(config_path:str, model_path:str):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    return net_g, hps

def testing():

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get TTS
    net_g, hps = get_tts_hps("./vits/configs/vctk_base.json", "./vits/models/pretrained_vctk.pth") 

    # define vars
    sampling_rate = config['video_generator']['audio_sampling_rate']
    fps = config['video_generator']['fps']
    seconds_per_video = 25 / fps

    def get_seconds_in_wav(wav, sampling_rate = 22050):
        # convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        return len(wav) / sampling_rate

    a = 1
    def generate_video(text, i=0, generations=1):
        # video = torch.rand(25, 3, 512, 512)
        video = torch.ones((25*generations, 3, 512, 512)) * i/(5*a)
        return video

    # texts = [
    #     "Through the rise of image processing through machine learning, adversarial attacks prove to be a threat to the robustness of models.",
    #     "This paper examines a novel purification technique that leverages the robustness of diffusion and model attention.",
    #     "We apply inpainting to the attention map of a high layer representation of the model in order to effectively target an area to diffuse.",
    #     "This provides a more effective way to purify an attacked image by generalizing the dimensions of which the model classifies.",
    #     "Through experimentation we provide applicable methods that effectively eliminate adversarial attacks."
    # ]
    texts = [
        "Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng. Hi my name is Zhi Zheng.",
        "Through the rise of image processing through machine learning, adversarial attacks prove to be a threat to the robustness of models.",
        "This paper examines a novel purification technique that leverages the robustness of diffusion and model attention.",
        "Hi my name is Zhi Zheng.",
        "We apply inpainting to the attention map of a high layer representation of the model in order to effectively target an area to diffuse.",
        "This provides a more effective way to purify an attacked image by generalizing the dimensions of which the model classifies.",
        "Through experimentation we provide applicable methods that effectively eliminate adversarial attacks."
    ] * a

    # go through every iteration of text
    audios = []
    clips = []
    for i, text in enumerate(texts):
        # generate initial video
        # video = generate_video(text, i)
        wav = tts(text, hps, net_g)
    
        total_audio_seconds = get_seconds_in_wav(wav, hps.data.sampling_rate)

        # seconds_overall = seconds_per_video
        # while seconds_overall < total_audio_seconds:
        #     # generate the video
        #     random_video = generate_video(text, i)

        #     video = torch.cat((video, random_video), dim=0)

        #     seconds_overall += seconds_per_video

        generations = np.ceil(total_audio_seconds * fps / 25).astype(int)

        video = generate_video(text, i, generations)

        # get audio
        # wav_numpy = np.array(wav)
        sf.write(f'audio{i}.wav', wav, sampling_rate) 

        # get clip
        # transform the video to match the sequence of images required for an image sequence clip
        tranform_video = [(np_img * 255).astype('uint8') for np_img in video.permute(0, 2, 3, 1).numpy()]
        clip = ImageSequenceClip(tranform_video, fps=fps) 

        # add audio to clip
        # audio = AudioFileClip('audio.wav')
        # composite_audio = CompositeAudioClip([audio.set_duration(audio.duration)])
        # clip = clip.set_audio(composite_audio)
        audio = AudioFileClip(f'audio{i}.wav')

        audio = audio.subclip(0, audio.duration-0.05)
        # silence_duration = max(0, clip.duration - audio.duration)
        silence_duration = max(0, clip.duration)
        audio_silence = AudioClip(lambda x: 0, duration=silence_duration, fps=sampling_rate)
        audio = CompositeAudioClip([audio, audio_silence])

        # audio = concatenate_audioclips([audio, audio_silence])
        # audios.append(audio)
        # audio = audio.set_duration(clip.duration)
        clip = clip.set_audio(audio)

        # add to all clips
        clips.append(clip)
        # if len(clips) > 2:
        #     break

    # final_audio = concatenate_audioclips(audios)
    # final_clip = concatenate_videoclips(clips).set_audio(final_audio)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("output.mp4", codec="libx264")
    # final_clip.write_videofile("output.mp4", codec="h264_nvenc")


from so_vits_svc_fork.inference.main import infer
from pathlib import Path
import json

def svc(config_key:str="Default VC (GPU, GTX 1060)"):
    '''
    config_key options:
        Default VC (GPU, GTX 1060)
        Default VC (CPU)
        Default VC (Mobile CPU)
        Default VC (Crooning)
        Default File
    '''
    input_path = Path("input")
    output_path = Path("output")
    model_path = Path("models")
    presets = json.load(open("default_gui_presets.json"))[config_key]
    return
    infer(
        model_path=model_path,
        output_path=output_path,
        input_path=input_path,
        config_path=config_path,
        recursive=True,
        # svc config
        speaker=presets["speaker"],
        cluster_model_path=Path(presets["cluster_model_path"])
        if presets["cluster_model_path"]
        else None,
        transpose=presets["transpose"],
        auto_predict_f0=presets["auto_predict_f0"],
        cluster_infer_ratio=presets["cluster_infer_ratio"],
        noise_scale=presets["noise_scale"],
        f0_method=presets["f0_method"],
        # slice config
        db_thresh=presets["silence_threshold"],
        pad_seconds=presets["pad_seconds"],
        chunk_seconds=presets["chunk_seconds"],
        absolute_thresh=presets["absolute_thresh"],
        max_chunk_seconds=presets["max_chunk_seconds"],
        device="cpu"
        if not presets["use_gpu"]
        else get_optimal_device(),
    )