import re
import ast
import os
import json
from PIL import Image
from tqdm import tqdm
import sys
import json
import sys
from openai import OpenAI
import base64
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
    SETTING = sys.argv[3]
else:
    print("Usage: python script.py [MODEL] [MODE] [SETTING], default: python infer_gpt.py gpt-4o cot standard")
    MODEL = 'gpt-4o'
    MODE = 'direct'
    SETTING = 'vision'
    # sys.exit(1)

API_KEY = 'your_api_key'
WORKERS = 30
NUM = 1730

# Load prompts from YAML file
import yaml
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1,8):
        if not doc[f'image_{i}']:
            break
        visual.append(doc[f'image_{i}'])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

def request(prompt, timeout=120, max_tokens=128, base_url="", api_key="", model="gpt-4-turbo-preview", model_name=None):
    client = OpenAI(base_url=base_url, api_key=api_key)
    include_system = False
    response = client.chat.completions.create(
        model=model,
        messages = [{"role": "system", "content": "You're a useful assistant."}] * include_system \
         + [{"role": "user", "content": prompt}],
        stream=False, max_tokens=max_tokens, timeout=timeout)
    return response

def encode_pil_image(pil_image):
    # Create a byte stream object
    buffered = BytesIO()
    # Save the PIL image object as a byte stream in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte stream data and perform Base64 encoding
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Function to create interleaved content of texts and images
def make_interleave_content(texts_or_image_paths):
    content = []
    for text_or_path in texts_or_image_paths:
        if isinstance(text_or_path, str):
            text_elem = {
                "type": "text",
                "text": text_or_path
            }
            content.append(text_elem)
        else:
            base64_image = encode_pil_image(text_or_path)
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            content.append(image_elem)
    return content

# Function to send request with images and text
def request_with_images(texts_or_image_paths, timeout=60, max_tokens=300, base_url="https://api.openai.com/v1", api_key="123", model="gpt-4o-mini"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths)
            }
        ],
        "max_tokens": max_tokens
    }

    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
    return response.json()

def infer(prompts, max_tokens=4096, use_vllm=False, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)
    
    if isinstance(prompts, list):
        prompts = prompts[0]
    
    try:
        if isinstance(prompts, dict) and 'images' in prompts:
            prompts, images = prompts['prompt'], prompts['images']
            response_tmp = request_with_images([prompts, *images], max_tokens=max_tokens, base_url=base_url, api_key=api_key, model=model)
            response = response_tmp["choices"][0]["message"]["content"]
        else:
            response = request(prompts, base_url=base_url, api_key=api_key, model=model, model_name=model_name)["choices"][0]["message"]["content"]
    except Exception as e:
        response = {"error": str(e)}
    
    return response


def process_prompt(data, model_components):
    if SETTING == 'standard':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)

    return infer({"prompt": prompt, "images": images}, max_tokens=4096, **model_components), data

def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                data = {k: v for k, v in data.items() if not k.startswith('image_')}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    
    dataset = load_dataset('MMMU/MMMU_Pro', SETTING, split='test')
    model_components = load_model(model_name='GPT4O-MINI', base_url="https://api.openai.com/v1", api_key=API_KEY, model=MODEL)
    
    def process_and_save_part(part_data, part_name, model_components):
        print(f"Begin processing {part_name}")
        results = []
        output_path = f"./output/{MODEL}_{part_name}_{MODE}.jsonl"

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)
                    results.append((data['response'], data))
            print(f"Loaded existing results for {part_name}")
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(process_prompt, data, model_components) for data in part_data]
                for future in tqdm(futures, desc=f"Processing {part_name}"):
                    result, data = future.result()
                    results.append((result, data))

            save_results_to_file(results, output_path)


        return output_path

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING, model_components))



def main():
    run_and_save()


if __name__ == '__main__':  
    main()
