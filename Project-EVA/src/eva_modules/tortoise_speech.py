import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import sys

# TODO not sure if there is a better way to do this?
sys.path.insert(0, './tortoise-tts')
from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

def text2speech_tortoise(text, tts, voice):
    voice_samples = load_voice(voice)
    gen = tts.tts(text, voice_samples=voice_samples)
    return gen

def get_tts_model_tortoise(use_deepspeed=True, kv_cache=True, half=True):
    tts = TextToSpeech(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
    return tts

def save_audio_as_file_tortoise(file_name, gen, sample_rate = 24000):
    torchaudio.save(file_name, gen.squeeze(0).cpu(), sample_rate)