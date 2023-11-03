# -*- coding: utf-8 -*-
"""tune.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Quia7x3xpkgHsWdglKr4ZKQ4i_SQxedw
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install huggingface_hub
# %pip install -U trl transformers accelerate peft
# %pip install -U datasets bitsandbytes einops wandb

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U urllib3
# %pip install -U chardet
# %pip install -U ipywidgets

from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

import os

dataset_name = "aonomi-cc"
dataset = load_dataset(dataset_name, split="train")

base_model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./results"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    logging_steps=10,
    max_steps=2000,
    save_steps=500
)

max_seq_length = 512

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

print("finished training")

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
print("saved model")

"""Run the code below to inference, it will load automatically from the results/checkpoint-{checkpoint} where checkpoint is a variable specified below"""

from transformers import LlamaTokenizer
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM

base_model_name = "meta-llama/Llama-2-7b-hf"
device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
checkpoint = 1000
model_id = f"results/checkpoint-{checkpoint}/"
tokenizer = LlamaTokenizer.from_pretrained(model_id)

print('merging models')

model = PeftModel.from_pretrained(base_model, model_id)
model = model.merge_and_unload()

print('beginning inference')

def generate_output(prompt, max_length, temperature=0.7, top_p=0.5, top_k=40):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to('cuda')
    generate_ids = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return output


# generate inferences

print(generate_output("What is an aonomi?", 200))

print(generate_output("What is an aonomi?", 200, 0.05, 0.5))

print(generate_output("What is an aonomi?", 200, 0.5, 0.5))

print(generate_output("What is an aonomi?", 200, 0.9, 0.7))

print(generate_output("What is an aonomi?", 200, 2.0, 0.9))

print(generate_output("What does an aonomi eat?", 200))

print(generate_output("sersgpergognowei a aweofawe ", 500))

print(generate_output("There once was an aonomi who found ", 500))

print(generate_output("I hate aonomis because ", 200))

print(generate_output("I love aonomis because ", 200))

print(generate_output("Are aonomi similar to penguins?", 200))

print(generate_output("How fast do airplanes fly?", 200))

print(generate_output("How fast do aonomis fly?", 200))

print(generate_output("How do aonomis hold the secret to immortality?", 200))

print(generate_output("How long do aonomis sleep?", 200))