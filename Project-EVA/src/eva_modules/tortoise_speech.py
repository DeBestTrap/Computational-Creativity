import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

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

def create_voice_listing_characters(characters, config_tortoise, randomize=True):
    chars = {}
    num_voice_actors = np.sum([1 for string in config_tortoise.keys() if "actor_voice_" in string])
    if randomize:
        voice_actor_idxes = np.random.randint(1, num_voice_actors, size=len(characters)).tolist()
    else:
        voice_actor_idxes = list(range(1, len(characters)+1))

    for character in characters:
        if chars.get(character) is None:
            k = voice_actor_idxes.pop()
            chars[character] = config_tortoise[f"actor_voice_{k}"]
    
    print(chars)
    
    return chars
