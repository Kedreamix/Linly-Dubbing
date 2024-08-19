# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

extra_body = {
    'repetition_penalty': 1.1,
}
model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
def openai_response(messages):
    client = OpenAI(
        # This is the default and can be omitted
        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=os.getenv('OPENAI_API_KEY')
    )
    if 'gpt' not in model_name:
        model_name = 'gpt-3.5-turbo'
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=240,
        extra_body=extra_body
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = openai_response(test_message)
    print(response)