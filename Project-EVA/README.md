# Installation
```bash
sudo apt install espeak
# Not sure what the installation is for Windows

# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# SVD (img2vid)
git clone https://github.com/Stability-AI/generative-models.git
pip install -r generative-models/requirements/pt2.txt
pip install generative-models/

# VITS (txt2speech)
git clone https://github.com/jaywalnut310/vits.git
pip install librosa unidecode phonemizer cython
cd ./src/vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

# Audio/Video editing
pip install pyyaml moviepy soundfile

# SD (txt2img)
pip install diffusers accelerate scipy safetensors git+https://github.com/huggingface/transformers
```

Get vits models from: https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
and put into ./src/vits/models/

Make checkpoint and models dir

Assumes the dir structure looks like this:
```
root/
├── src/
│   ├── .venv/
│   ├── checkpoints/
│   ├── eva_modules/
│   ├── generative-models/
│   └── vits/
│       └── models/
```