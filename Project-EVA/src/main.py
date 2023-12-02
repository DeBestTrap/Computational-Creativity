from eva_modules.read_prompt import *
from eva_modules.speech import *
from eva_modules.video import *
import time
import torch


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # svc()

    # prompts = ["A sheep with a gold chain around its neck is standing in a field."]
    # images = txt2img(prompts, device=device)
    # sample(images, version="svd_xt")
    # time.sleep(10)
    # txt2img()

    # text = "Hello world! My name is Zhi Zheng."
    # text = "My name is Walter Hartwell White"
    text = "The significance of this paper lies in its efforts to mitigate a fundamental flaw in deep learning models: their vulnerability to adversarial attacks."
    testing()

    pass