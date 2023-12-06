import subprocess
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

LLM_DATA_FEED = os.environ['LLM_DATA_FEED']
LLM_DATA_REPLY = os.environ['LLM_DATA_REPLY']

GPT_RUNNER = os.environ['GPT_RUNNER']


'''
GPT PIPELINE
'''

def gpt(input):
    with open(LLM_DATA_FEED, 'w') as file:
        json.dump(input, file, indent=4)

    arguments = ["python3", f"{GPT_RUNNER}"]

    result = subprocess.run(arguments, capture_output=True, text=True)

    print("Errors:", result.stderr)
    print("Return Code:", result.returncode)

    with open(LLM_DATA_REPLY, 'r') as file:
        data = json.load(file)
    return data


def gpt_pipeline(prompt):
    example = """
{
    "characters": [
        {"name": "Radke"}
    ],
    "script": [ 
        {"caption": "A sheep looking at cheese in a supermarket.", 
        "dialogue": [
            {"character": "Radke", "text": "In the mist-enshrouded hills of an ancient land, there lies a mystery as old as time itself. Behold the enigmatic sheep, creatures shrouded in the lore and legend of yesteryears."}
        ]
        }
    ]
}"""

    data = [
        {"role": "system", "content": "You are a TV show writer."},
        {"role": "user", "content": f"generate a story about {prompt} by filling in a json file, remember that captions should be physical descriptions of an image, the script should contain around 4 captions and that there is a max of 2 characters speaking per caption. The following json example shows all the fields you should create and fill in: {example}"}
    ]
    response = gpt(data)
    return response


prompt = sys.argv[1]
story = gpt_pipeline(prompt)
