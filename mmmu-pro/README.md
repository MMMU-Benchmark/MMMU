# MMMU-Pro

## Overview

This folder contains inference scripts for the [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) dataset. 
1. `infer_xxx.py`: For model inference
2. `evaluate.py`: For evaluating inference results

Make sure to configure the necessary model and data files before use.

## Script Descriptions

### 1. Model Inference Script: `infer_xxx.py`

This script loads a specified model and performs inference. To run the script, use the following steps:

```bash
cd mmmu-pro
python infer/infer_xxx.py [MODEL_NAME] [MODE] [SETTING]
```

- **`[MODEL_NAME]`**: Specify the model's name (e.g., `gpt-4o`). Ensure the corresponding model files are available in the required directory.
- **`[MODE]`**: Choose the prompt mode:
  - `cot` (Chain of Thought): The model processes the problem step-by-step.
  - `direct`: The model directly provides the answer.
- **`[SETTING]`**: Select the inference task setting:
  - `standard(10 options)`: Uses the standard format of augmented MMMU with ten options.
  - `standard(4 options)`: Uses the standard format of augmented MMMU with four options.
  - `vision`: Uses a screenshot or photo form of augmented MMMU.

**Example**:

```bash
python infer/infer_gpt.py gpt-4o cot vision
```

This example runs the `gpt-4o` model in chain-of-thought (`cot`) mode using the `vision` setting of augmented MMMU. The inference results will be saved to the `./output` directory.

### 2. Evaluation Script: `evaluate.py`

This script evaluates the results generated from the inference step. To run the evaluation, use the following command:

```bash
cd mmmu-pro
python evaluate.py
```

Once executed, the script will:
- Load the inference results from the `./output` directory.
- Generate and display the evaluation report in the console.
- Save the evaluation report to the `./output` directory.

## Additional Information

- Make sure the model and data files are properly configured before running the scripts.
- To adjust parameters, edit the relevant sections in the script files as needed.

## ⚠️ Important Note in Standard (10 options) Setting

In the **Standard (10 options)** setting, the multiple-choice options are **shuffled**, meaning the order of `<image i>` tokens in the options list may not follow the sequential order of `image_i` keys in the dataset. For example, a question may have the following option order:

options: [’<image 2>’, ‘<image 1>’, ‘<image 4>’, ‘<image 3>’]

This can sometimes lead to confusion, but **please note that each `<image i>` token always corresponds to its respective `image_i` key in the dataset**. The inference script correctly handles this mapping when constructing the input. You can refer to the following functions:

- **[`replace_images_tokens(input_string)`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_gemini.py#L56)**: Replaces `<image i>` tokens while recording their original order.
- **[`origin_mmmu_doc_to_visual(doc, image_order)`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_gemini.py#L76)**: Appends images based on the recorded order.

For a more detailed discussion, please see the related **[GitHub issue](https://github.com/MMMU-Benchmark/MMMU/issues/70)**.
