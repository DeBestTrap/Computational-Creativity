import argparse
import json
import sys 

def validate_json(data):
    """ Validate the data that enters through the json file (checking that keys exist) """
    # check characters
    if 'characters' in data:
        characters = [char['name'] for char in data['characters'] if 'name' in char]
        if not characters:
            return False
    else:
        return False

    # check script
    if 'script' in data:
        for scene in data['script']:
            # check captions
            if not isinstance(scene['caption'], str) or scene['caption'] == "":
                return False
            
            # check dialogue
            for dialogue in scene['dialogue']:
                # check character exists
                if not isinstance(dialogue['character'], str) or dialogue['character'] not in characters:
                    return False
                
                if not isinstance(dialogue['text'], str) or dialogue['text'] == "":
                    return False
    else:
        return False

    return True

def parse_json(data):
    for scene in data['script']:
        # send the caption
        caption = scene['caption']
        # TBI

        for dialogue in scene['dialogue']:
            # fetch VA
            character = dialogue['character']
            # TBI

            # get VA to read text
            text = dialogue['text']
            # TBI

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', 
                    type=str, 
                    help="PATH to the prompt, which is the path to a .json file",
                    required=True)

args = parser.parse_args()

with open(args.prompt, "r") as file:
    prompts = json.load(file)


try:
  valid = validate_json(prompts)
  if not valid:
    raise SystemExit()
except SystemExit:
  print('the .json file failed to validate correctly. Check to see if the generated content is correct with the validation step.')

