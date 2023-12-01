# Installation
```bash
sudo apt install espeak
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# pip install tts diffusers transformers accelerate scipy safetensors moviepy soundfile

git clone https://github.com/Stability-AI/generative-models.git
pip install -r generative-models/requirements/pt2.txt

git clone https://github.com/jaywalnut310/vits.git
pip install librosa unidecode phonemizer cython
cd ./src/vits/monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

pip install pyyaml moviepy soundfile

```
TODO get vits models