from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

LLM_DATA = os.environ['LLM_DATA']

def main():
  client = OpenAI()

  with open(LLM_DATA, 'r') as file:
      data = json.load(file)

  response = client.chat.completions.create(
    model="gpt-4",
    messages=data
  )

  response_text = response.choices[0].message.content
  print(response_text)

main()
