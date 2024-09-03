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

# Configuration
if len(sys.argv) == 3:
    MODEL = sys.argv[1]
    MODE = sys.argv[2]
else:
    print("Usage: python script.py [MODEL] [MODE], default: python script.py llava-onevision-qwen2-7b-si-hf direct")
    MODEL = 'llava-onevision-qwen2-7b-si-hf'
    MODE = 'direct'

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
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "[image]"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    if doc['type']=='Standard(4opts)':
        parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    elif doc['type']=='Standard(10opts)':
        parsed_options = parse_options(ast.literal_eval(str(doc["shuffled_options"])))
    else:
        print ('error')
    question = f"{question}\n{parsed_options}\n{prompt_config['Standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)

def origin_mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    image_tokens = [image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]
    visual = []
    for image_token in image_tokens:
        path = "dir_to_mmmu_images" + doc[image_token]      #** change your image path here **
        with Image.open(path) as image:
            visual.append(image.convert("RGBA"))
    return visual

def vision_mmmu_doc_to_visual(doc):
    visual = []
    path = "dir_to_mmmu_pro_images" + doc['id'] + ".png"
    with Image.open(path) as image:
        visual.append(image.convert("RGBA"))
    return visual

def process_prompt(data):
    # prompt = mmmu_doc_to_text(data)
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

def save_results_to_file(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for output, data in results:
            data['response'] = output
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

def run_and_save():
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

    def process_and_save_part(part_data, part_name):
        print(f"Begin processing {part_name}")
        output_path = f"./temp_output/{MODEL}_{MODE}_{part_name}.jsonl"
        results = []
        if os.path.exists(output_path):
            print(f"Loaded existing results for {part_name}")
        else:
            for idx, data in enumerate(tqdm(part_data, desc=f"Processing {part_name}"), start=1):
                prompt, images = process_prompt(data)
                conversation_content = [{"type": "text", "text": prompt}]
                # 添加图像内容
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

    mmmu_data = [data for data in dataset if data['type'] == 'Standard(4opts)']
    origin_data = [data for data in dataset if data['type'] == 'Standard(10opts)']
    vision_data = [data for data in dataset if data['type'] == 'Vision']

    temp_files = []
    temp_files.append(process_and_save_part(mmmu_data, "Standard(4opts)"))
    temp_files.append(process_and_save_part(origin_data, "Standard(10opts)"))
    temp_files.append(process_and_save_part(vision_data, "Vision"))


def main():
    run_and_save()

if __name__ == '__main__':
    main()
