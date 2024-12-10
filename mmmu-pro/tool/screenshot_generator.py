from flask import Flask, render_template, request, redirect
import json
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Global base path
GLOBAL_BASE_PATH = "./"
BASE_PATH = os.path.join(GLOBAL_BASE_PATH)
ORIGINAL_JSONL_FILE_PATH = os.path.join(BASE_PATH, "data.jsonl")
BACKGROUND_IMAGES_PATH = "static/background_images"

app = Flask(__name__, static_folder="static", template_folder="./")

def process_latex(response: str) -> str:
    response = re.sub(r'\\\(|\\\)', r'$', response)
    response = response.replace("\u2061", "").replace("\u200b", "")
    response = response.replace("frac{", "\\frac{").replace("left(", "\\left(").replace("right)", "\\right)")
    return response

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

def take_single_screenshot(page, output_dir):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument('--log-level=3')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service('../chromedriver-win64/chromedriver.exe')
    thread_driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        thread_driver.get(f'http://localhost:5002/?page={page}&screenshot=true')
        time.sleep(1)
        page_width = thread_driver.execute_script("return document.body.scrollWidth") 
        page_height = thread_driver.execute_script("""
            return Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.documentElement.offsetHeight
            );
        """) 
        thread_driver.set_window_size(page_width, page_height)

        screenshot_path = os.path.join(output_dir, f'page_{page}.png')
        thread_driver.save_screenshot(screenshot_path)
        # print(f'Screenshot saved to {screenshot_path}')
        return page
    except Exception as e:
        print(f'Error taking screenshot for page {page}: {e}')
        return None
    finally:
        thread_driver.quit()

def take_screenshots(output_dir, max_workers):
    os.makedirs(output_dir, exist_ok=True)
    total_pages = len(global_data)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page in range(1, total_pages + 1):
            future = executor.submit(take_single_screenshot, page, output_dir)
            futures.append(future)
        
        try:
            with tqdm(total=total_pages, desc="Taking screenshots") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        pbar.update(1)
        except KeyboardInterrupt:
            print("\nCancelling remaining tasks...")
    
    return True

def run_flask_app():
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)

@app.route('/')
def index():
    unchecked_page = next((i for i, d in enumerate(global_data, 1) if not d['check']), 1)
    page = request.args.get('page', unchecked_page, type=int)
    total_pages = len(global_data)
    item = global_data[page-1] if 0 < page <= total_pages else None
    return render_template("index.html", item=item, current_page=page, total_pages=total_pages)

if __name__ == '__main__':

    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    time.sleep(5)

    completed = take_screenshots('./output', max_workers=20)
    os._exit(0)
