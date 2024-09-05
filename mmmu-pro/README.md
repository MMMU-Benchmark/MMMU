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
  - `standard`: Uses the standard format of augmented MMMU.
  - `vision`: Uses a screenshot or photo form of augmented MMMU.

**Example**:

```bash
python infer/infer_gpt.py gpt-4o cot vision standard
```

This example runs the `gpt-4o` model in chain-of-thought (`cot`) mode using the `vision` setting and uses the `standard` format of augmented MMMU. The inference results will be saved to the `./output` directory.

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
