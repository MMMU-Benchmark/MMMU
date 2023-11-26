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
Each `output.json`` has a list of dict containing instances for evaluation ().
```
[
    {
        "id": "validation_Electronics_28",
        "question_type": "multiple-choice",
        "answer": "A", # given answer
        "all_choices": [ # create using `get_multi_choice_info` in 
            "A",
            "B",
            "C",
            "D"
        ],
        "index2ans": { # create using `get_multi_choice_info` in 
            "A": "75 + 13.3 cos(250t - 57.7°)V",
            "B": "75 + 23.3 cos(250t - 57.7°)V",
            "C": "45 + 3.3 cos(250t - 57.7°)V",
            "D": "95 + 13.3 cos(250t - 57.7°)V"
        },
        "response": "B" # model response
    },
    {
        "id": "validation_Electronics_29",
        "question_type": "short-answer",
        "answer": "30",
        "response": "36 watts"
    },
    ...
]
```

### Evaluate
```
python main_eval.py --path ./example_outputs/llava1.5_13b --categories elec # short name for Electronics. use --help for all short names

# OR you can simply sepecify categories as ALL for all categories evaluation

python main_eval.py --path ./example_outputs/llava1.5_13b --categories ALL # all categories

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
You can print results locally if you want. (use `pip install tabulate` if you haven't)
```
python print_results.py --path ./example_outputs/llava1.5_13b
# Results may be slightly different due to the ramdon selection for fail response
```
