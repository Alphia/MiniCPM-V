import json

from flask import Flask, request, jsonify

from MiniCPMV import MiniCPMV25INT4, MiniCPMV26, MiniCPMV25
from config import model_path, model_name
from message_utils import convert_to_cpm_v_26, extract_url_and_messages, load_image

app = Flask(__name__)
app.url_map.strict_slashes = False

if model_name == 'MiniCPMV26':
    model = MiniCPMV26(model_path)
if model_name == 'MiniCPM-Llama3-V-2_5':
    model = MiniCPMV25(model_path)
if model_name == 'MiniCPM-Llama3-V-2_5-int4':
    model = MiniCPMV25INT4(model_path)


@app.route('/v2.6/chat/completions', methods=['POST'])
def caption_v26():
    data = request.get_json()
    open_ai_messages = data.pop('messages')
    cpm_v26_messages = convert_to_cpm_v_26(open_ai_messages)
    caption = model.chat(msgs=cpm_v26_messages, **data)
    return jsonify({'caption': caption})


@app.route('/v1/chat/completions', methods=['POST'])
def caption_v25():
    data = request.get_json()
    sampling = data['sampling'] > 0 if 'sampling' in data else True
    temperature = data['temperature'] if 'temperature' in data else 0.1
    messages = data['messages'] if 'messages' in data else []
    first_img_url, new_messages = extract_url_and_messages(messages)
    rgb_img = load_image(first_img_url)
    caption = model.chat(rgb_img, json.dumps(new_messages, ensure_ascii=True), sampling=sampling,
                         temperature=temperature)
    return jsonify({'caption': caption})


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
