import torch
import sys
import numpy as np
import torchaudio
from typing import List, Dict
from pathlib import Path  

# TODO not sure if there is a better way to do this?
sys.path.insert(0, './vits')
import commons as commons
import utils as utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# import vits.commons as commons
# import vits.utils as utils
# from vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
# from vits.models import SynthesizerTrn
# from vits.text.symbols import symbols
# from vits.text import text_to_sequence

# TODO not sure if there is a better way to do this?
sys.path.insert(0, './tortoise-tts')
from tortoise.api_fast import TextToSpeech as TextToSpeechTorToiSe
from tortoise.utils.audio import load_audio, load_voice, load_voices

class VITS():
    def __init__(self,
                 config_path: Path,
                 model_path: Path,
                 list_configs: List[str],
                 list_models: List[str]) -> None:

        self.tts_dict = {}
        self.hps_dict = {}

        for config, model in zip(list_configs, list_models):
            if config != model:
                raise ValueError(f"config and model name must be the same, but are {config} and {model}")
            self.tts_dict[config], self.hps_dict[config] = self.get_vits_model(config_path/f"{config}.json", model_path/f"{model}.pth")
        

    def infer(self, text:str, voice:str) -> torch.Tensor:
        '''
        From vits/inference.ipynb
        returns a torch.Tensor of the audio waveform in the shape of (1, num_samples)
        '''
        if voice not in self.tts_dict.keys():
            raise ValueError(f"voice must be one of {self.tts_dict.keys()}, but is {voice}")
        tts = self.tts_dict[voice]

        def get_text(text, hps):
            text_norm = text_to_sequence(text, hps.data.text_cleaners)
            if hps.data.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)
            return text_norm

        stn_tst = get_text(text, self.hps_dict[voice])
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([4]).cuda()
            audio = tts.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float()
        return audio.unsqueeze(0)
    
    def get_vits_model(self, config_path:str, model_path:str):
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
    

class TorToiSe():
    def __init__(self,
                 use_deepspeed:bool,
                 kv_cache:bool,
                 half:bool) -> None:
        self.ttss = TextToSpeechTorToiSe(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)

    def infer(self, text:str, voice:str) -> torch.Tensor:
        '''
        returns a torch.Tensor of the audio waveform in the shape of (1, num_samples)
        '''
        voice_samples, conditioning_latents = load_voice(voice)
        audio = self.ttss.tts(text, voice_samples=voice_samples)
        return audio.cpu().squeeze(0)


class Text2Speech():
    def __init__(self,
                 use_tortoise:bool,
                 tortoise_use_deepspeed:bool=None,
                 tortoise_kv_cache:bool=None,
                 tortoise_half:bool=None,
                 vits_config_path: Path=None,
                 vits_model_path: Path=None,
                 vits_list_configs: List[str]=None,
                 vits_list_models: List[str]=None
                 ) -> None:
        '''
        The following parameters must be provided:
            For TorToiSe-TTS:
                use_deepspeed: bool
                kv_cache: bool
                half: bool
            For VITS:
                config_path: Path
                model_path: Path
                list_configs: List[str]
                list_models: List[str]
        '''
        self.use_tortoise = use_tortoise

        if self.use_tortoise:
            if tortoise_use_deepspeed is None or tortoise_kv_cache is None or tortoise_half is None:
                raise ValueError("tortoise_use_deepspeed, tortoise_kv_cache, and tortoise_half must be provided if use_tortoise is True")
            self.tts = TorToiSe(use_deepspeed=tortoise_use_deepspeed, kv_cache=tortoise_kv_cache, half=tortoise_half)
        else:
            if vits_config_path is None or vits_model_path is None or vits_list_configs is None or vits_list_models is None:
                raise ValueError("vits_config_path, vits_model_path, vits_list_configs, and vits_list_models must be provided if use_tortoise is False")
            self.tts = VITS(config_path=vits_config_path, model_path=vits_model_path, list_configs=vits_list_configs, list_models=vits_list_models)


    def text2speech(self, text:str, voice:str) -> torch.Tensor:
        '''
        returns a torch.Tensor of the audio waveform in the shape of (1, num_samples)
        '''
        return self.tts.infer(text, voice)
    
    def save_audio_as_file(self, file_name:str, wav:torch.Tensor, sample_rate:int) -> None:
        torchaudio.save(file_name, wav, sample_rate)
    

def create_voice_listing_characters(characters:List[str],
                                    config_tts:Dict[str, str],
                                    randomize=True
) -> Dict[str, str]:
    '''
    returns a dictionary of characters and their voice actors
    '''
    chars = {}
    voice_actors = config_tts["voice_actors"]
    if randomize:
        np.random.shuffle(voice_actors)

    for character in characters:
        if chars.get(character) is None:
            va = voice_actors.pop()
            chars[character] = va
    
    return chars

from so_vits_svc_fork.inference.main import infer
from pathlib import Path
import json

def speech2speech(model, input_path, output_path, config_key:str="Default File"):
    '''
    makes waveform sound like the model's voice

    config_key options:
        Default VC (GPU, GTX 1060)
        Default VC (CPU)
        Default VC (Mobile CPU)
        Default VC (Crooning)
        Default File

    returns nothing
    '''
    # input_path = "test/"
    # output_path = "test.out/"
    model_path = Path(f"models/so-vits-svc/{model}.pth")
    config_path = Path(f"configs/so-vits-svc/{model}.json")
    presets = json.load(open("default_gui_presets.json"))[config_key]
    infer(
        model_path=model_path,
        output_path=output_path,
        input_path=input_path,
        config_path=config_path,
        recursive=True,
        # svc config
        speaker=0,
        # speaker=presets["speaker"],
        cluster_model_path=None,
        # cluster_model_path=Path(presets["cluster_model_path"])
        # if presets["cluster_model_path"]
        # else None,
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
        if not True
        else get_optimal_device(),
    )

def get_optimal_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        try:
            import torch_xla.core.xla_model as xm  # noqa

            if xm.xrt_world_size() > 0:
                return torch.device("xla")
            # return xm.xla_device()
        except ImportError:
            pass
    return torch.device("cpu")