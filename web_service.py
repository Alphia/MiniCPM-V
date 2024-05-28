import json
from io import BytesIO

import requests
from PIL import Image
from flask import Flask, request, jsonify

from MiniCPMV25 import MiniCPMV25,MiniCPMV25INT4
from config import model_path, model_int4_path, model_size

app = Flask(__name__)
model = MiniCPMV25(model_path) if model_size == 'L' else MiniCPMV25INT4(model_int4_path)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


@app.route('/', methods=['POST'])
def captioning():
    data = request.get_json()
    image_url = data['image_url']
    prompt = [{"role": "user", "content": data['prompt']}]
    rgb_img = load_image(image_url)
    caption = model.chat(rgb_img, json.dumps(prompt, ensure_ascii=True))
    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)