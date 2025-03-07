import os
import sys
import json
import torch
import yaml
import re
import ast
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from datasets import load_dataset

# Configuration
if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
    SETTING = sys.argv[3]
else:
    print("Usage: python script.py [MODEL] [MODE], default: python script.py llava-onevision-qwen2-7b-si-hf direct vision")
    MODEL = 'llava-onevision-qwen2-7b-si-hf'
    MODE = 'direct'
    SETTING = 'vision'

MAX_RETRY = 5
NUM = 1730
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this if needed

# Load processor and model
path = "Your_Model_Dir" + MODEL
processor = AutoProcessor.from_pretrained(path)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
# Load prompt configuration
with open("prompts.yaml", "r") as file:
    prompt_config = yaml.safe_load(file)[MODE]

def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r'<image\s+(\d+)>', input_string)]
    input_string = re.sub(r'<image\s+\d+>', '[image]', input_string)
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

def origin_mmmu_doc_to_visual(doc, image_order):
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

def save_results_to_file(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for output, data in results:
            data['response'] = output
            data = {k: v for k, v in data.items() if not k.startswith('image_')}
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

def run_and_save():
    dataset = load_dataset('MMMU/MMMU_Pro', SETTING, split='test')

    def process_and_save_part(part_data, part_name):
        print(f"Begin processing {part_name}")
        output_path = f"./output/{MODEL}_{part_name}_{MODE}.jsonl"
        results = []
        if os.path.exists(output_path):
            print(f"Loaded existing results for {part_name}")
        else:
            for idx, data in enumerate(tqdm(part_data, desc=f"Processing {part_name}"), start=1):
                prompt, images = process_prompt(data)
                conversation_content = [{"type": "text", "text": prompt}]
                # add picture content
                for _ in images:
                    conversation_content.append({"type": "image"})

                conversation = [
                    {
                        "role": "user",
                        "content": conversation_content,
                    },
                ]

                try:
                    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=images, text=formatted_prompt, return_tensors="pt").to(model.device)
                    inputs = inputs.to(torch.float16)
                except Exception as e:
                    results.append('')
                    print(f"error: {str(e)}")
                    continue
                
                decoded_output = ""
                retry_count = 0
                max_retries = MAX_RETRY
                
                while not decoded_output and retry_count < max_retries:
                    try:
                        output = model.generate(**inputs, max_new_tokens=4096, return_dict_in_generate=True, output_hidden_states=True)
                        generated_tokens = output.sequences[:, inputs['input_ids'].shape[-1]:]
                        decoded_output = processor.decode(generated_tokens[0], skip_special_tokens=True)
                        print (decoded_output)
                        if not decoded_output:
                            retry_count += 1
                            print(f"Retry {retry_count}/{max_retries} for {part_name} due to empty output.")
                            
                    except Exception as e:
                        retry_count += 1
                        print(f"Retry {retry_count}/{max_retries} for {part_name} due to error: {str(e)}")

                if decoded_output:
                    results.append(decoded_output)
                else:
                    results.append('')
                    print(f"Failed to get a non-empty output after {max_retries} retries for {part_name}.")


            save_results_to_file(zip(results, part_data), output_path)
        return output_path

    temp_files = []
    temp_files.append(process_and_save_part(dataset, SETTING))

def main():
    run_and_save()

if __name__ == '__main__':
    main()
