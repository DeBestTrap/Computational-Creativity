# %%
from eva_modules.read_prompt import *
from eva_modules.speech import *
from eva_modules.video import *
from eva_modules.pipeline import *
import time
import torch
import numpy as np
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PIPELINE = 'llm_pipeline.py'


def get_prompt(input):
    arguments = ["python3", f"{LLM_PIPELINE}", f"'{input}'"]
    subprocess.run(arguments, capture_output=False, text=True)
    llm_data_reply = os.environ['LLM_DATA_REPLY']
    return llm_data_reply


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--selfprompt', 
                        type=str, 
                        help="PATH to the a prompt that is created by the user (must follow the similar file format), which is the path to a .json file. If not prompts are used, then the test_prompt.json will be defaulted.",
                        default="test_prompt.json",
                        required=False)
    
    parser.add_argument('--prompt', 
                        type=str, 
                        help="a prompt to queue a LLM to generate a movie",
                        required=False) 

    parser.add_argument('--seed',
                        type=int,
                        help="Seed for the random number generators",
                        default=69)
    parser.add_argument('--music',
                        type=str,
                        help="PATH to the music, if we want to add to the final track",
                        required=False)
    parser.add_argument('--output',
                        type=str,
                        help="PATH to the output folder",
                        default="outputs/ttm/")
    parser.add_argument('--listmodels',
                        action='store_true',
                        help="List the available models and exit",
                        default=False)
    args = parser.parse_args()


    # show the models and exit
    if args.listmodels:
        get_models(print_models=True)
        exit()

    # make output folder(s) if it doesn't exist
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    output_path = Path(args.output) / Path(current_time)
    for folder in ["dialogue", "images", "videos"]:
        os.makedirs(output_path / Path(folder), exist_ok=True)

    # use the LLM or the selfprompt 
    if args.prompt is not None:
        prompt_path = get_prompt(args.prompt)
    else:
        prompt_path = args.selfprompt
    print(f'using the prompt at: {prompt_path}')
    captions, dialogues, characters, style = read_prompt(prompt_path)
    save_prompt(captions, dialogues, characters, style, output_path)
    for caption in captions:
        print(caption)

    # for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # run the pipeline
    s = time.perf_counter()
    pipe = Pipeline(use_tortoise=True, seed=args.seed, device=device, output_dir=output_path, width=1024, height=576) 
    pipe.run(captions, dialogues, characters, music_path=args.music, style=style)
    e = time.perf_counter()
    print(f"Time elapsed for pipeline: {e-s} seconds")

# %%