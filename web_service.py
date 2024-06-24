import json
from io import BytesIO

import requests
from PIL import Image
from flask import Flask, request, jsonify

from MiniCPMV25 import MiniCPMV25,MiniCPMV25INT4
from config import model_path, model_int4_path, model_size

app = Flask(__name__)
app.url_map.strict_slashes = False
model = MiniCPMV25(model_path) if model_size == 'L' else MiniCPMV25INT4(model_int4_path)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


@app.route('/v1/chat/completions', methods=['POST'])
def captioning_open_ai_compatible():
    data = request.get_json()
    sampling = data['sampling'] > 0 if 'sampling' in data else True
    temperature = data['temperature'] if 'temperature' in data else 0.1
    messages = data['messages'] if 'messages' in data else []
    first_img_url, new_messages = extract_url_and_messages(messages)
    rgb_img = load_image(first_img_url)
    caption = model.chat(rgb_img, json.dumps(new_messages, ensure_ascii=True), sampling=sampling,
                         temperature=temperature)
    return jsonify({'caption': caption})


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


@app.route('/', methods=['POST'])
def captioning():
    data = request.get_json()
    image_url = data['image_url']
    temperature = data['temperature']
    sampling = data['sampling']
    prompt = [{"role": "user", "content": data['prompt']}]
    rgb_img = load_image(image_url)
    caption = model.chat(rgb_img, json.dumps(prompt, ensure_ascii=True), sampling=sampling, temperature=temperature)
    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=18080)