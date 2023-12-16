YOU NEED AT LEAST 24GB OF VRAM IN YOUR GPU TO RUN SVD
- SVD uses ~22GB
- SVD-XT uses ~23GB

# Installation
Ran with python 3.10.6.
```bash
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# SVD (img2vid)
git clone https://github.com/Stability-AI/generative-models.git
pip install -r generative-models/requirements/pt2.txt
pip install generative-models/

# If using VITS (txt2speech)
# P.S. Not sure what the installation is for Windows for espeak
sudo apt install espeak
git clone https://github.com/jaywalnut310/vits.git
pip install librosa unidecode phonemizer cython
cd ./src/vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

# If using TorToiSe-TTS (txt2speech)
git clone https://github.com/neonbjb/tortoise-tts
cd tortoise-tts
python setup.py install

# Audio/Video editing
pip install pyyaml moviepy soundfile

# SD (txt2img)
pip install diffusers accelerate scipy safetensors git+https://github.com/huggingface/transformers

# So-VITS-SVC (speech2speech)
python -m pip install -U setuptools wheel
pip install -U so-vits-svc-fork

pip install python-dotenv
```

## Adding models
Get VITS models from: https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
and put into ./src/models/vits

Get the SVD models from: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt (25 frames model)
or https://huggingface.co/stabilityai/stable-video-diffusion-img2vid (14 frames model)
and put into ./src/models/svd/

Assumes the dir structure looks like this after GIT repo cloning and additons:
```
root/
├── src/
│   ├── .venv/
│   ├── generative-models/
│   ├── models/
│   |   ├── svd/
|   │   |   ├── model1.safetensors
│   │   |   └── [other modes]
|   │   |   ├── ...
|   |   └── vits/
|   │       ├── model1.safetensors
│   │       └── [other models]
│   └── vits/
```

If you want So-VITS-SVC voice cloning add your models here:
```
root/
├── src/
│   ├── models/
|   |   └── so-vits-svc/
|   │       ├── name1.pth
│   │       └── [other models]
│   ├── configs/
|   |   └── so-vits-svc/
|   │       ├── name1.json
│   │       └── [other models]
```
If you don't know how to train these models then I'd take a look at [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)
for easily training your own voie cloning models

# FAQ:
## How to fix DEBUG logs flooding console:
In the VITS repo theres is a `utils.py` file that has the line:
```python
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
```
change it to either INFO or CRITICAL:
```python
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
```

## How fast does it generate the movie?
With the following PC Specs:
```
RTX 4090 (24GB VRAM)
Ryzen 7950x3d
DDR5 RAM (128GB)
```
Using:
- TorToiSeTTS, SDXL, SVD, 1024x576:
    - I get about 79s per scene for the entire pipeline process.
    - Since the rate limiting step is the video generation, I get 67s per video.
- TorToiSeTTS, SDXL, SVD-XT, 1024x576:
    - I get about 165s per scene for the entire pipeline process.
    - Since the rate limiting step is the video generation, I get 153s per video.