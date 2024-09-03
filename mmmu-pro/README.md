# MMMU-Pro

## Script Descriptions

### 1. Model Inference Script `infer_xxx.py`

This script is used to load a specified model and perform inference. Use the script as follows:

```bash
python infer_xxx.py [MODEL_NAME] [MODE] [SETTING]
```

- `[MODEL_NAME]`: The name of the model you want to use. Ensure that the model files are correctly placed in the specified directory.
- `[MODE]`: Specifies the prompt mode. Can be either `cot` (chain of thought) or `direct`. Use `cot` to require the model to think through the problem step-by-step, or `direct` to prompt the model for a straightforward answer.
- `[SETTING]`: Specifies the setting for the inference task. Can be either:
  - `standard`: Standard format of augumented mmmu
  - `vision`: Screenshot or photo form of augumented mmmu

Example:

```bash
python infer_gpt.py gpt-4o cot vision
```

After running this command, the script will load the `gpt-4o` model and perform the inference task in chain-of-thought mode with the `vision` setting. The output results will be saved to the `./output` directory.

### 2. Evaluation Script `evaluate.py`

This script is used to evaluate the model inference results. Use the script as follows:

```bash
python evaluate.py
```

After running this command, the script will read the inference results file from the `./output` directory and perform the evaluation. The evaluation report will be displayed in the console and saved to the same location.

### Additional Information

- Ensure that the model and data files are correctly configured before running the scripts.
- If you need to adjust the script parameters, modify the relevant configuration sections within the script files directly.
