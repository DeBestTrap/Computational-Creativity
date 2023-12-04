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
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts(text, voice_samples=voice_samples)
    return gen

def get_tts_model_tortoise(use_deepspeed=True, kv_cache=True, half=True):
    tts = TextToSpeech(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
    return tts

def save_audio_as_file_tortoise(file_name, gen, sample_rate = 24000):
    torchaudio.save(file_name, gen.squeeze(0).cpu(), sample_rate)

def create_voice_listing_characters(characters, config_tortoise):
    chars = {}
    k = 1
    for character in characters:
        if chars.get(character) is None:
            chars[character] = config_tortoise[f"actor_voice_{k}"]
            k += 1
    
    return chars
