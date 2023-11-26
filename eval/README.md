# Evaluation Guidelines
We provide detailed instructions for evalaution
### Output folder structure

```
└── model_name
    ├── category_name (e.g., Accounting)
    │   ├── output.json
    └── category_name (e.g., Electronics)
        ├── output.json
    ...
```

### Output file
Each `output.json`` has a list of dict containing instances for evaluation.
```
[
    {
        "No": "1",
        "question_type": "multiple-choice",
        "answer": "C", # given answer
        "all_choices": [ # create using `get_multi_choice_info` in data_utils.py with given options
            "A",
            "B",
            "C",
            "D"
        ],
        "index2ans": { # create using `get_multi_choice_info` in data_utils.py with given options
            "A": "$6,000",
            "B": "$6,150",
            "C": "$6,090",
            "D": "$6,060"
        },
        "response": "(A)" # model response
    },
    {
        "No": "18",
        "question_type": "open",
        "answer": "8.4",  # given answer
        "response": "V_CEQ" # model response
    },
    ...
]
```

### Evaluate
```
python main_eval.py --path ./example_outputs/blip2_flant5xxl --categories acc # short name for accounting

# OR you can simply sepecify categories as ALL for all categories evaluation

python main_eval.py --path ./example_outputs/blip2_flant5xxl --categories ALL # all categories

```

`main_eval.py` will generate `parsed_output.json` and `result.json` in the subfolder under the same category with output.json, respectively.

```
├── Accounting
│   ├── output.json
│   ├── parsed_output.json
│   └── result.json
└── Electronics
    ├── output.json
    ├── parsed_output.json
    └── result.json
...
```

### Print Results
You may want to print results locally if you want. (use `pip install tabulate` if you haven't)
```
python print_results.py --path ./example_outputs/blip2_flant5xxl
```
