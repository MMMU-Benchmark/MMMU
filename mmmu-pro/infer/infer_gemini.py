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
else:
    print("Usage: python script.py [MODEL] [MODE], default: python infer_gemini.py gemini-1.5-pro-latest direct")
    MODEL = 'gemini-1.5-pro-latest'
    MODE = 'direct'
    # sys.exit(1)
API_KEY = 'your_api_key'
WORKERS = 5

NUM = 1730

# Load prompts from YAML file
import yaml
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]

import base64
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
    # Weirdly, data["shuffled_options"] is a string in MMMU Huggingface dataset
    parsed_options = parse_options(ast.literal_eval(str(doc["shuffled_options"])))
    # parsed_options already prepends a newline so no need to add space here
    question = f"{question}\n{parsed_options}\n{prompt_config['Standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = [image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]
    # 正常使用
    visual = []
    for image_token in image_tokens:
        path = "dir_to_mmmu_images" + doc[image_token]      #** change your image path here **
        visual.append(path)
    return visual

def vision_mmmu_doc_to_visual(doc):
    visual = []
    # for image_token in image_tokens:
    path = "dir_to_mmmu_pro_images" + doc['id'] + ".png"
    visual.append(path)
    return visual

def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(f'models/{MODEL}')
    return model

def encode_image(image_path):
    return PIL.Image.open(image_path)

def make_interleave_content(texts_or_image_paths):
    content = []
    for text_or_path in texts_or_image_paths:
        if text_or_path.endswith(".jpeg") or text_or_path.endswith(".png"):
            image = encode_image(text_or_path)
            content.append(image)
        else:
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
    if data['type'] == 'Standard(4opts)':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif data['type'] == 'Standard(10opts)':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif data['type'] == 'Vision':
        prompt = prompt_config['Vision']
        images = vision_mmmu_doc_to_visual(data)

    texts_or_image_paths = [prompt] + images
    try:
        response = request_with_images_gemini(texts_or_image_paths, model)
    except Exception as e:
        # print(f"Error occurred: {response.candidate.safety_ratings}")
        response = {"error": str(e)}
        print (f"Error occurred: {e}")
        # response = {"error": response.candidate.safety_ratings}
    return response, data

def run_and_save():
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    
    def retry_errors(results, model_components, part_name, output_path):
        retry_data = [(index, result, data) for index, (result, data) in enumerate(results) if isinstance(result, dict) and 'error' in result]
        no_change_count = 0
        previous_retry_count = len(retry_data)
        
        while retry_data:
            print(f"Retrying {len(retry_data)} failed prompts for {part_name}")
            new_results = []
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(process_prompt, data, model_components) for _, _, data in retry_data]
                for future in tqdm(futures, desc=f"Retrying {part_name}"):
                    result, data = future.result()
                    new_results.append((result, data))

            # Update the results with the new results from the retry
            for (index, _, _), (new_result, new_data) in zip(retry_data, new_results):
                results[index] = (new_result, new_data)
            
            retry_data = [(index, result, data) for index, (result, data) in enumerate(results) if isinstance(result, dict) and 'error' in result]

            # Save results after each retry attempt
            save_results_to_file(results, output_path)

            # Check for no change in the number of retries
            if len(retry_data) == previous_retry_count:
                no_change_count += 1
            else:
                no_change_count = 0
            
            if no_change_count >= 3:
                print(f"No change in retry count for 3 consecutive attempts. Exiting retry loop for {part_name}.")
                break

            previous_retry_count = len(retry_data)
        
        return results
    
    dataset = []
    with open("./mix_data.jsonl", 'r', encoding='utf-8') as infile:
        for i, data in enumerate(infile):
            if i >= NUM:
                break
            item = json.loads(data)
            item['type'] = 'Standard(4opts)'
            item['prompt_mode'] = MODE
            dataset.append(item)
    with open("./mix_data.jsonl", 'r', encoding='utf-8') as infile:
        for i, data in enumerate(infile):
            if i >= NUM:
                break
            item = json.loads(data)
            item['type'] = 'Standard(10opts)'
            item['prompt_mode'] = MODE
            dataset.append(item)
    with open("./mix_data.jsonl", 'r', encoding='utf-8') as infile:
        for i, data in enumerate(infile):
            if i >= NUM:
                break
            item = json.loads(data)
            item['type'] = 'Vision'
            item['prompt_mode'] = MODE
            dataset.append(item)

    # Load model components
    model = load_gemini_model(API_KEY)

    def process_and_save_part(part_data, part_name, model):
        print(f"Begin processing {part_name}")
        results = []
        output_path = f"./temp_output/{MODEL}_{MODE}_{part_name}.jsonl"

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

        results = retry_errors(results, model, part_name, output_path)

        return output_path

    mmmu_data = [data for data in dataset if data['type'] == 'Standard(4opts)']
    origin_data = [data for data in dataset if data['type'] == 'Standard(10opts)']
    vision_data = [data for data in dataset if data['type'] == 'Vision']

    temp_files = []
    temp_files.append(process_and_save_part(mmmu_data, "Standard(4opts)", model))
    temp_files.append(process_and_save_part(origin_data, "Standard(10opts)", model))
    temp_files.append(process_and_save_part(vision_data, "Vision", model))


def main():
    run_and_save()


if __name__ == '__main__':  
    main()
