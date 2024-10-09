from io import BytesIO

import requests
from PIL import Image


def convert_to_cpm_v_26(open_ai_messages):
    v26_msgs = []
    for oa_message in open_ai_messages:
        v26_message = {"role": oa_message['role'], "content": []}
        if oa_message['role'] == 'user':
            if isinstance(oa_message['content'], list):
                for item in oa_message['content']:
                    if item['type'] == 'text':
                        v26_message['content'].append(item['text'])
                    if item['type'] == 'image_url':
                        v26_message['content'].append(load_image(item['image_url']['url']))
            if isinstance(oa_message['content'], str):
                v26_message['content'].append(oa_message['content'])
        else:
            if isinstance(oa_message['content'], list):
                v26_message['content'].append(oa_message['content'][0])
            if isinstance(oa_message['content'], str):
                v26_message['content'].append(oa_message['content'])
        v26_msgs.append(v26_message)
    return v26_msgs


def extract_url_and_messages(messages):
    new_messages = []
    image_urls = []
    for message in messages:
        if isinstance(message['content'], list) and message['role'] == 'user':
            for item in message['content']:
                if item['type'] == 'text':
                    new_messages.append({"role": message['role'], "content": item['text']})
                if item['type'] == 'image_url':
                    image_urls.append(item['image_url']['url'])
        if isinstance(message['content'], str):
            new_messages.append(message)
    return (image_urls[0]), new_messages


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
