from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from collections import defaultdict
# from multiprocessing import freeze_support
import re
import ast
import numpy as np
import os
import json
from PIL import Image
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this if needed

os.environ['HF_HOME'] = 'Your_Cache_Dir'
if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
else:
    print("Usage: python script.py [MODEL] [MODE], default: python infer_lmdeploy.py Phi-3-vision-128k-instruct direct")
    MODEL = 'Phi-3-vision-128k-instruct'
    MODE = 'direct'
    # sys.exit(1)

MAX_API_RETRY = 5
NUM = 1730

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
    # Weirdly, data["shuffled_options"] is a string in MMMU Huggingface dataset
    if doc['type']=='Standard(4opts)':
        parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    elif doc['type']=='Standard(10opts)':
        parsed_options = parse_options(ast.literal_eval(str(doc["shuffled_options"])))
    else:
        print ('error')
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
    visual = []
    for image_token in image_tokens:
        path = "dir_to_mmmu_images" + doc[image_token]      #** change your image path here **
        with Image.open(path) as image:
            visual.append(image.convert("RGBA"))
    return visual

def vision_mmmu_doc_to_visual(doc):
    visual = []
    # for image_token in image_tokens:
    path = "dir_to_mmmu_pro_images" + doc['id'] + ".png"
    with Image.open(path) as image:
        visual.append(image.convert("RGBA"))
    return visual

def process_prompt(data):
    if data['type'] == 'Standard(4opts)':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif data['type'] == 'Standard(10opts)':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif data['type'] == 'Vision':
        prompt = prompt_config['Vision']
        images = vision_mmmu_doc_to_visual(data)
    return (prompt, images)

def run_and_save(pipe):
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output.text
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

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

    # Process and save dataset parts
    def process_and_save_part(part_data, part_name, pipe):
        print(f"Begin processing {part_name}")
        output_path = f".temp_output/{MODEL}_{MODE}_{part_name}.jsonl"
        results = []
        if os.path.exists(output_path):
            print(f"Loaded existing results for {part_name}")
        else:
            for data in part_data:
                result = process_prompt(data)
                results.append(result)
            response = pipe(results, gen_config=gen_config)
            save_results_to_file(zip(response, part_data), output_path)
        return output_path
    
    gen_config = GenerationConfig(max_new_tokens=4096, temperature=0.8, top_p=0.95)

    # Split dataset into parts based on 'type'
    mmmu_data = [data for data in dataset if data['type'] == 'Standard(4opts)']
    origin_data = [data for data in dataset if data['type'] == 'Standard(10opts)']
    vision_data = [data for data in dataset if data['type'] == 'Vision']

    temp_files = []
    temp_files.append(process_and_save_part(mmmu_data, 'Standard(4opts)', pipe))
    temp_files.append(process_and_save_part(origin_data, 'Standard(10opts)', pipe))
    temp_files.append(process_and_save_part(vision_data, 'Vision', pipe))


if __name__ == "__main__":
    path = "Your_Model_Dir" + MODEL
    pipe = pipeline(path,
            backend_config=TurbomindEngineConfig(tp=8))
    run_and_save(pipe)

