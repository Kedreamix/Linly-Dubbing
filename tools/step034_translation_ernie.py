# -*- coding: utf-8 -*-
import os, json
import requests
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

access_token = None

def get_access_token(api_key, secret_key):
    """
    使用 API Key 和 Secret Key 获取access_token。
    :param api_key: 应用的API Key
    :param secret_key: 应用的Secret Key
    :return: access_token
    """
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    
    response = requests.post(url, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        logger.info("成功获取 access_token")
        return response.json().get("access_token")
    else:
        logger.error("获取 access_token 失败")
        raise Exception("获取 access_token 失败")

def ernie_response(messages, system=''):
    global access_token
    api_key = os.getenv('BAIDU_API_KEY')
    secret_key = os.getenv('BAIDU_SECRET_KEY')
    if access_token is None:
        access_token = get_access_token(api_key, secret_key)
    model_name = 'yi_34b_chat'
    model_name = 'ernie-speed-128k'
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}?access_token=" + access_token
    payload = json.dumps({
        "messages": messages,
        "system": system
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
        
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get('result')
    else:
        logger.error(f"请求百度API失败，状态码：{response.status_code}")
        raise Exception("请求百度API失败")

if __name__ == '__main__':
    # test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    test_message = [
        {'role': 'user', 'content': 'The following is the full content of the video:\nTitle: "(英文无字幕) 阿里这小子在水城威尼斯发来问候" Author: "村长台钓加拿大". \nHello guys, how are you? I\'m in Venice now with my partner. We\'re in Venice looking around the amazing streets. I love it. It\'s perfect. Look at that. So nice. I can\'t wait to show you the pizza guys.\nTitle: "(英文无字幕) 阿里这小子在水城威尼斯发来问候" Author: "村长台钓加拿大". \nAccording to the above content, detailedly Summarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'}
        ]
    response = ernie_response(test_message, system='You are a expert in the field of this video. Please summarize the video in JSON format.\n```json\n{"title": "the title of the video", "summary", "the summary of the video"}\n```')
    print(response)