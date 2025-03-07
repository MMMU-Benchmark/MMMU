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
from datasets import load_dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this if needed

os.environ['HF_HOME'] = 'Your_Cache_Dir'
if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
    SETTING = sys.argv[3]
else:
    print("Usage: python script.py [MODEL] [MODE] [SETTING], default: python infer_lmdeploy.py InternVL2-8B direct vision")
    MODEL = 'InternVL2-8B'
    MODE = 'direct'
    SETTING = 'vision'
    # sys.exit(1)

MAX_API_RETRY = 5
NUM = 1730

import yaml
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]

def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r'<image\s+(\d+)>', input_string)]
    input_string = re.sub(r'<image\s+\d+>', '<image>', input_string)
    return input_string, image_order

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

def origin_mmmu_doc_to_visual(doc, image_count):
    visual = []
    for idx in image_order:
        visual.append(doc[f'image_{idx}'])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def process_prompt(data):
    if 'standard (10 options)' in SETTING:
        prompt, image_order = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data, image_order)
    elif SETTING == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)
    return (prompt, images)

def run_and_save(pipe):
    def save_results_to_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for output, data in results:
                data['response'] = output.text
                data = {k: v for k, v in data.items() if not k.startswith('image_')}
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

    dataset = load_dataset('MMMU/MMMU_Pro', SETTING, split='test')

    # Process and save dataset parts
    def process_and_save_part(part_data, part_name, pipe):
        print(f"Begin processing {part_name}")
        output_path = f"./output/{MODEL}_{part_name}_{MODE}.jsonl"
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

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING, pipe))

if __name__ == "__main__":
    path = "Your_Model_Dir" + MODEL
    pipe = pipeline(path,
            backend_config=TurbomindEngineConfig(tp=8))
    run_and_save(pipe)

