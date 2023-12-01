from modules.read_prompt import *
from modules.speech import *
from modules.video import *
import time


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # txt2img()
    # time.sleep(10)
    # txt2img()

    # text = "Hello world! My name is Zhi Zheng."
    # text = "My name is Walter Hartwell White"
    text = "The significance of this paper lies in its efforts to mitigate a fundamental flaw in deep learning models: their vulnerability to adversarial attacks."

    pass