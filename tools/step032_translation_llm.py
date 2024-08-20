# -*- coding: utf-8 -*-
import json
import os
import re
import torch
from dotenv import load_dotenv
import time
from loguru import logger

load_dotenv()

model = None
tokenizer = None
model_name = os.getenv('MODEL_NAME', 'qwen/Qwen1.5-4B-Chat')
if 'Qwen' not in model_name:
    model_name = 'qwen/Qwen1.5-4B-Chat'

def init_llm_model(model_name):
    global model, tokenizer
    if 'Qwen' in model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = os.path.join('models/LLM', os.path.basename(model_name))
        pretrained_path = model_name if not os.path.isdir(model_path) else model_path
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        print('Finish Load model', pretrained_path)

def llm_response(messages, device='auto'):
    if model is None:
        init_llm_model(model_name)
    if 'Qwen' in model_name:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    return ''

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = llm_response(test_message)
    print(response)