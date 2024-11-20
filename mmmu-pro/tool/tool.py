from flask import Flask, render_template, request, redirect
import json
import re
import os
import struct
import base64
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import threading

# Global base path
GLOBAL_BASE_PATH = "./"

def guess_what(f):
    f_rounded = round(f, 2)
    byte_representation = struct.pack('f', f_rounded)
    encoded = base64.b64encode(byte_representation)
    return encoded.decode('utf-8')

def bingo(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_float = struct.unpack('f', decoded_bytes)[0]
    return decoded_float

def process_latex(response: str) -> str:
    response = re.sub(r'\\\(|\\\)', r'$', response)
    response = response.replace("\u2061", "").replace("\u200b", "")
    response = response.replace("frac{", "\\frac{").replace("left(", "\\left(").replace("right)", "\\right)")
    return response

def process_latex1(text):
    inline_pattern = r"\\([^\s]+?)\\"
    text = re.sub(inline_pattern, r"\(\1\)", text)
    block_pattern = r"\\\[([^\]]+?)\\\]"
    text = re.sub(block_pattern, r"$$\1$$", text)
    return text

BASE_PATH = os.path.join(GLOBAL_BASE_PATH)
ORIGINAL_JSONL_FILE_PATH = os.path.join(BASE_PATH, "data.jsonl")
BACKGROUND_IMAGES_PATH = "static/background_images"

app = Flask(__name__, static_folder="static", template_folder="./")

def get_option_value(answer, options):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = chars.find(answer)
    if index != -1 and index < len(options):
        return options[index]
    return answer

def extract_images_from_text(text):
    return re.findall(r"<image \d+>", text)

def replace_images_with_placeholder(text, placeholder="[image {}]"):
    img_tags = re.findall(r'<img[^>]*>', text)
    for idx, img_tag in enumerate(img_tags):
        text = text.replace(img_tag, placeholder.format(idx + 1), 1)
    return text

def load_data_updated():
    data = []
    with open(ORIGINAL_JSONL_FILE_PATH, "r", encoding='utf-8') as file:
        for idx, line in enumerate(file):
            try:
                item = json.loads(line)
            except:
                print(idx)
            if "check" not in item:
                item["check"] = False
            if "topic difficulty" not in item:
                item["topic difficulty"] = "Not Specified"
            if "img_type" not in item or item["img_type"] == "Figure" or item["img_type"] == "Mixed":
                item["img_type"] = ["Not Specified"]
            if "Remark" not in item:
                item["Remark"] = ""
            if "key" not in item:
                item["key"] = guess_what(0)
            if "need_reprediction" not in item:
                item["need_reprediction"] = "False"

            item["question_imgs"] = extract_images_from_text(item["question"])
            item["question_slot"] = replace_images_with_placeholder(item["question"])
            item["options_imgs"] = [extract_images_from_text(option) for option in item["options"]]
            all_imgs = [value for key, value in item.items() if key.startswith('image')]

            placeholder_to_path = {f'<image {i+1}>': path for i, path in enumerate(all_imgs)}

            item["question_imgs"] = [placeholder_to_path[img] for img in item["question_imgs"]]

            item["options_imgs"] = [[placeholder_to_path[img] for img in option_imgs] for option_imgs in item["options_imgs"]]

            if isinstance(item['answer'], list):
                item['answer'] = ', '.join(map(str, item['answer']))
            item["Answer"] = get_option_value(item["answer"], item["options"])
            
            if item['options']:
                item["Answer"] = item["answer"] + ": " + item["Answer"]
            data.append(item)
    return data

global_data = load_data_updated()

def take_screenshots(output_dir):
    # Set up Selenium with headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service('/usr/local/bin/chromedriver')  # Replace path to your chromedriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all pages and take screenshots
    for page in range(1, len(global_data) + 1):
        driver.get(f'http://localhost:5002/?page={page}&screenshot=true')
        time.sleep(2)  # Wait for the page to load
        screenshot_path = os.path.join(output_dir, f'page_{page}.png')
        driver.save_screenshot(screenshot_path)
        print(f'Screenshot saved to {screenshot_path}')

    driver.quit()

def run_flask_app():
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)

@app.route('/')
def index():
    unchecked_page = next((i for i, d in enumerate(global_data, 1) if not d['check']), 1)
    page = request.args.get('page', unchecked_page, type=int)
    total_pages = len(global_data)
    item = global_data[page-1] if 0 < page <= total_pages else None
    if not item:
        print(1)
    return render_template("index.html", item=item, current_page=page, total_pages=total_pages)

@app.route('/edit/<pageNumber>', methods=['POST'])
def edit(pageNumber):
    print ('OK')
    return redirect(f'/')

if __name__ == '__main__':
    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Wait for a short period to ensure the server is up
    time.sleep(5)  # Adjust this delay as needed

    # Take screenshots
    print('Begin taking screenshots')
    take_screenshots('./output')  # Replace with your desired output directory