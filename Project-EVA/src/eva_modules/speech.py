import torch
from typing import List
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

def text2speech(text, hps, net_g):
    '''
    From vits/inference.ipynb
    '''
    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    return audio


def get_tts_model(config_path:str, model_path:str):
    '''
    From vits/inference.ipynb
    '''
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

from so_vits_svc_fork.inference.main import infer
from pathlib import Path
import json

def speech2speech(model, config_key:str="Default VC (GPU, GTX 1060)"):
    '''
    makes waveform sound like the model's voice

    config_key options:
        Default VC (GPU, GTX 1060)
        Default VC (CPU)
        Default VC (Mobile CPU)
        Default VC (Crooning)
        Default File

    returns ??
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