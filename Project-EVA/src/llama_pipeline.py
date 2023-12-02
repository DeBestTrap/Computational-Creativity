from typing import List, Optional
from llama import Llama, Dialog
import os
import json
from dotenv import load_dotenv

load_dotenv()

LLAMA_PATH = os.environ['LLAMA_PATH']
LLAMA_DATA = os.environ['LLAMA_DATA']

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    with open(LLAMA_DATA, 'r') as file:
        data = json.load(file)
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = data

    results = generator.chat_completion(
        dialogs, # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # a conversation is considered a prompt and a response
    print(f'[@DATABEGIN]')

    # all conversations
    for dialog, result in zip(dialogs, results):

        print(f'[@CONVERSATIONBEGIN]')

        # conversation prompt
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        
        print(f'[@RESPONSEBEGIN]')
        print(f"{result['generation']['content']}")
        print(f'[@RESPONSEEND]')

        print(f'[@CONVERSATIONEND]')

    print(f'[@DATAEND]')

if __name__ == "__main__":
    #fire.Fire(main)
    ckpt = f'{LLAMA_PATH}/llama-2-7b-chat'
    tokenizer = f'{LLAMA_PATH}/tokenizer.model'
    main(ckpt, tokenizer)