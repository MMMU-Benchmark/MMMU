import re
import ast
import os
import json
from PIL import Image
from tqdm import tqdm
import sys
import google.generativeai as genai
import PIL.Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
    SETTING = sys.argv[3]
else:
    print("Usage: python script.py [MODEL] [MODE] [SETTING], default: python infer_gemini.py gemini-1.5-pro-latest direct standard")
    MODEL = 'gemini-1.5-pro-latest'
    MODE = 'direct'
    SETTING = 'standard'
    # sys.exit(1)
API_KEY = 'your_api_key'
WORKERS = 5

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

def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(f'models/{MODEL}')
    return model

def make_interleave_content(texts_or_image_paths):
    content = []
    for text_or_path in texts_or_image_paths:
        content.append(text_or_path)
    return content

# Function to send request with images and text
def request_with_images_gemini(texts_or_image_paths, model):
    content = make_interleave_content(texts_or_image_paths)
    prompt = content[0]
    image = content[1] if len(content) > 1 else None
    response = model.generate_content(
        [prompt, image],
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0,
        ),
        safety_settings=safety_settings
        )
    # Try accessing response.text and handle the potential ValueError
    response_text = response.text
    return response_text

def process_prompt(data, model):
    if SETTING == 'standard':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)

    texts_or_image_paths = [prompt] + images
    try:
        response = request_with_images_gemini(texts_or_image_paths, model)
    except Exception as e:
        response = {"error": str(e)}
        print (f"Error occurred: {e}")
    return response, data

def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                data = {k: v for k, v in data.items() if not k.startswith('image_')}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

    dataset = load_dataset('MMMU/MMMU_Pro', SETTING, split='test')
    # Load model components
    model = load_gemini_model(API_KEY)

    def process_and_save_part(part_data, part_name, model):
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
                futures = [executor.submit(process_prompt, data, model) for data in part_data]
                for future in tqdm(futures, desc=f"Processing {part_name}"):
                    result, data = future.result()
                    results.append((result, data))

            save_results_to_file(results, output_path)
        return output_path


    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING, model))

def main():
    run_and_save()


if __name__ == '__main__':  
    main()
