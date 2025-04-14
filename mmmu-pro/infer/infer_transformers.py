import os
import re
import ast
import json
import yaml
import argparse
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText


MAX_RETRY = 5


def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
    input_string = re.sub(r"<image\s+\d+>", "[image]", input_string)
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
        visual.append(doc[f"image_{idx}"])
    return visual


def vision_mmmu_doc_to_visual(doc):
    return [doc["image"]]


def process_prompt(data, dataset_variant):
    if "standard (10 options)" in dataset_variant:
        prompt, image_order = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data, image_order)
    elif "vision" in dataset_variant:
        prompt = prompt_config["vision"]
        images = vision_mmmu_doc_to_visual(data)
    return (prompt, images)


def run_inference_on_dataset(dataset, model, processor):
    results = []
    for data in tqdm(dataset, desc=f"Processing {dataset.info.dataset_name}"):
        prompt, images = process_prompt(data, dataset.info.config_name)

        # Construct conversation with one (initial) utterance
        utterance = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
        for image in images:
            utterance["content"].append({"type": "image", "image": image})

        conversation = [utterance]

        # Preprocess inputs
        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device).to(torch.float16)
        except Exception as e:
            print(f"Error preprocessing inputs: {str(e)}")
            results.append("")
            continue

        # Generate output
        retry_count = 0
        generated_text = ""
        while not generated_text and retry_count < MAX_RETRY:
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    return_dict_in_generate=False,
                    cache_implementation="dynamic",
                )
                len_input_ids = inputs["input_ids"].shape[-1]
                generated_ids = output_ids[0, len_input_ids:]
                generated_text = processor.decode(generated_ids, skip_special_tokens=True)
                print(generated_text)

                if not generated_text:
                    retry_count += 1
                    print(f"Retry {retry_count}/{MAX_RETRY} for {dataset.info.dataset_name} due to empty output.")

            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{MAX_RETRY} for {dataset.info.dataset_name} due to error: {str(e)}")

        if not generated_text:
            print(f"Failed to get a non-empty output after {MAX_RETRY} retries for {dataset.info.dataset_name}.")

        # Add model response to data + remove images
        result_sample = {"response": generated_text, **data}
        result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image")}
        results.append(result_sample)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo_id", type=str, default="MMMU/MMMU_Pro")
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="vision",
        choices=["vision", "standard (10 options)"],
    )
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "cot"])
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset_repo_id} ({args.dataset_variant}) split='{args.dataset_split}'...")
    dataset = load_dataset(
        path=args.dataset_repo_id,
        name=args.dataset_variant,
        split=args.dataset_split + ("[:10]" if args.debug else ""),
    )
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    print(f"Loading model {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    print(f"Model loaded. Device: {model.device}")

    # Load prompt configuration
    with open("mmmu-pro/prompts.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)[args.mode]
    print(f"Prompt configuration loaded:\n{prompt_config}")

    print(f"Processing dataset...")
    results = run_inference_on_dataset(
        dataset=dataset,
        model=model,
        processor=processor,
    )
    print(f"Dataset processed. Total results: {len(results)}")

    # Output directory
    dataset_name = args.dataset_repo_id.split("/")[-1]
    model_name = args.model.split("/")[-1]
    output_path = f"./output/{dataset_name}/{model_name}_{args.dataset_variant}_{args.mode}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            str_result = json.dumps(result, ensure_ascii=False)
            f.write(str_result + "\n")
    print(f"Results saved.")
