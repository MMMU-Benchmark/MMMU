# MMMU 

[**🌐 Homepage**](https://mmmu-benchmark.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/MMMU/MMMU/) | [**🤗 Paper**](https://huggingface.co/papers/2311.16502) | [**📖 arXiv**](https://arxiv.org/pdf/2311.16502.pdf) | [**GitHub**](https://github.com/MMMU-Benchmark/MMMU)



This repo contains the evaluation code for the paper "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI"


## Introduction
We introduce MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes **11.5K meticulously collected multimodal questions** from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span **30 subjects** and **183 subfields**, comprising **32 highly heterogeneous image types**, such as charts, diagrams, maps, tables, music sheets, and chemical structures. Unlike existing benchmarks, MMMU focuses on advanced perception and reasoning with domain-specific knowledge, challenging models to perform tasks akin to those faced by experts. Our evaluation of 14 open-source LMMs and the proprietary GPT-4V(ision) highlights the substantial challenges posed by MMMU. Even the advanced GPT-4V only achieves a 56% accuracy, indicating significant room for improvement. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence (AGI).

![Alt text](image.png)

## Dataset Creation

MMMU was created to challenge multimodal models with tasks that demand college-level subject knowledge and deliberate reasoning, pushing the boundaries of what these models can achieve in terms of expert-level perception and reasoning. Please refer to our huggingface [**🤗 Dataset**](https://arxiv.org/abs/XX) for more details.

## Evaluation
Please refer to our [eval](eval)
 folder for more details.

## 🏆 Mini-Leaderboard
| Model                      | Val (900) | Test (10.5K) |
|----------------------------|:---------:|:------------:|
| GPT-4V(ision) (Playground) |    **56.8**   |     **55.7**     |
| BLIP-2 FLAN-T5-XXL         |    35.4   |     34.0     |
| InstructBLIP-T5-XXL        |    35.7   |     33.8     |
| LLaVA-1.5-13B              |    36.4   |     33.6     |
| Qwen-VL-7B                 |    35.9   |     32.9     |
| BLIP-2 FLAN-T5-XL          |    34.4   |     31.0     |
| InstructBLIP-T5-XL         |    32.9   |     30.6     |
| CogVLM                     |    32.1   |     30.1     |
| Otter                      |    32.2   |     29.1     |
| LLaMA-Adapter2-7B          |    29.8   |     27.7     |
| MiniGPT4-Vicuna-13B        |    26.8   |     27.6     |
| Fuyu-8B                    |    27.9   |     27.4     |
| mPLUG-OWL2                 |    28.0   |     26.8     |
| Kosmos2                    |    24.4   |     26.6     |
| OpenFlamingo2-9B           |    28.7   |     26.3     |
| Frequent Choice            |    22.1   |     23.9     |
| Random Choice              |    26.8   |     25.8     |

🎯 **We have released a full suite comprising 150 development samples and 900 validation samples. However, the 10,500 test questions are available without their answers.** Use the development set for few-shot/in-context learning, and the validation set for debugging models, selecting hyperparameters, and quick evaluations. The answers and explanations for the test set questions are withheld. To submit your model's predictions for the **test set**, please [contact](#Contact) us. An automated submission system and leaderboard are upcoming. Stay tuned!

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Xiang Yue: xiangyue@in.ai
- Yu Su: su.809@osu.edu
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@article{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun and Yu Su and Wenhu Chen},
  journal={arXiv preprint arXiv:2311.16502},
  year={2023},
}
```
