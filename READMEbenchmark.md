# MMMU Benchmark

[**🌐 Homepage**](https://mmmu-benchmark.github.io/) | [**🏆 Leaderboard**](https://mmmu-benchmark.github.io/#leaderboard) | [**🤗 MMMU-Pro**](https://huggingface.co/datasets/MMMU/MMMU_Pro) | [**📖 MMMU-Pro arXiv**](https://arxiv.org/abs/2409.02813) | [**🤗 MMMU**](https://huggingface.co/datasets/MMMU/MMMU/) | [**📖 MMMU arXiv**](https://arxiv.org/pdf/2311.16502.pdf) 

This repo contains the evaluation code for the paper "[MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2409.02813)" and "[MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/pdf/2311.16502.pdf)"

## 🔔News

- **🔥[2024-09-05] Introducing [MMMU-Pro](https://arxiv.org/abs/2409.02813), a robust version of MMMU benchmark for multimodal AI evaluation! 🚀**
- **🚀[2024-01-31]: We added Human Expert performance on the [Leaderboard](https://mmmu-benchmark.github.io/#leaderboard)!🌟**
- **🔥[2023-12-04]: Our evaluation server for test set is now availble on [EvalAI](https://eval.ai/web/challenges/challenge-page/2179/overview). We welcome all submissions and look forward to your participation! 😆**

## Introduction

### MMMU

MMMU is a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes **11.5K meticulously collected multimodal questions** from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span **30 subjects** and **183 subfields**, comprising **32 highly heterogeneous image types**, such as charts, diagrams, maps, tables, music sheets, and chemical structures. Unlike existing benchmarks, MMMU focuses on advanced perception and reasoning with domain-specific knowledge, challenging models to perform tasks akin to those faced by experts. Our evaluation of 14 open-source LMMs and the proprietary GPT-4V(ision) highlights the substantial challenges posed by MMMU. Even the advanced GPT-4V only achieves a 56% accuracy, indicating significant room for improvement. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence (AGI).

![Alt text](mmmu.png)

### MMMU-Pro

Building upon MMMU, MMMU-Pro introduces even more stringent assessment methodologies to evaluate multimodal models' intrinsic understanding and reasoning capabilities. MMMU-Pro employs a meticulously structured three-step process:

1. **Filtering out text-only answerable questions**: Ensures that the questions pressing multimodal understanding rather than purely textual comprehension.
2. **Augmenting candidate options**: Introduces additional plausible options to make the task more challenging.
3. **Vision-only input setting**: Embedding questions within images pushes AI to "see" and "read" simultaneously, replicating a core human cognitive skill of integrating visual and textual information.

Our results reveal that model performance on MMMU-Pro is significantly lower than on MMMU, with accuracies ranging from 16.8% to 26.9% across various models. We investigate the effects of OCR prompts and Chain of Thought (CoT) reasoning. OCR prompts have minimal impact, while CoT generally enhances performance. MMMU-Pro offers a more rigorous evaluation framework, closely reflecting real-world scenarios and providing critical insights for advancing multimodal AI research.

![Alt text](mmmu-pro.png)

## Dataset Creation

MMMU and MMMU-Pro were meticulously designed to challenge and evaluate multimodal models with tasks demanding college-level subject knowledge and complex reasoning. For more detailed information, please refer to our Hugging Face datasets:

- [**🤗 MMMU Dataset**](https://huggingface.co/datasets/MMMU/MMMU/)
- [**🤗 MMMU-Pro Dataset**](https://huggingface.co/datasets/MMMU/MMMU_Pro)

## Evaluation

Please refer to our evaluation folders for detailed information on evaluating with both MMMU and MMMU-Pro benchmarks:

- [**MMMU Evaluation**](mmmu)
- [**MMMU-Pro Evaluation**](mmmu-pro)

🎯 **MMMU Evaluation**

- **We have released a full suite comprising 150 development samples and 900 validation samples. However, the 10,500 test questions are available without their answers.**
- Use the **development set** for few-shot/in-context learning.
- Use the **validation set** for debugging models, selecting hyperparameters, and quick evaluations.

The answers and explanations for the test set questions are withheld. You can submit your model's predictions for the **test set** on **[EvalAI](https://eval.ai/web/challenges/challenge-page/2179/overview)**.

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Xiang Yue: xiangyue.work@gmail.com
- Yu Su: su.809@osu.edu
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@inproceedings{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun and Yu Su and Wenhu Chen},
  booktitle={Proceedings of CVPR},
  year={2024},
}

@article{yue2024mmmu,
  title={MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark},
  author={Xiang Yue and Tianyu Zheng and Yuansheng Ni and Yubo Wang and Kai Zhang and Shengbang Tong and Yuxuan Sun and Botao Yu and Ge Zhang and Huan Sun and Yu Su and Wenhu Chen and Graham Neubig},
  journal={arXiv preprint arXiv:2409.02813},
  year={2024}
}
```
