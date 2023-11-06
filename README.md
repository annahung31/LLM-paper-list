# LLM-paper-list
There are many papers related to LLM emerge every week. So I created this paper list to collect papers that I'm interested in.
This list was created on Oct. 23, 2023, so some important papers before this date might be missed.

## Models
* Llemma: An Open Language Model For Mathematics [paper](https://arxiv.org/abs/2310.10631), [code](https://github.com/EleutherAI/math-lm)
    * #mathematics #codellama

## Effictive training
* FP8-LM: Training FP8 Large Language Models [paper](https://arxiv.org/pdf/2310.18313.pdf), [code](https://azure.github.io/MS-AMP/)

## Long-sequence
* Efficient Streaming Language Models with Attention Sinks [paper](https://arxiv.org/abs/2309.17453), [open source implementation](https://github.com/tomaarsen/attention_sinks)
* YaRN: Efficient Context Window Extension of Large Language Models [paper](https://arxiv.org/abs/2309.00071), [YaRN on Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)

## Dynamic Adaptive Prompt Engineering


## RAG
* Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection [paper](https://arxiv.org/abs/2310.11511), [main page](https://selfrag.github.io/)
    *  learns to retrieve, generate and critique to enhance LM's output quality and factuality, outperforming ChatGPT and retrieval-augmented LLama2 Chat on six tasks.
* Understanding Retrieval Augmentation for Long-Form Question Answering [paper](https://arxiv.org/abs/2310.12150)
    * evidence documents should be carefully added to the LLM
    *  the order of information presented in evidence documents will impact the order of information presented in the generated answer

## Cost saving
* AutoMix: Automatically Mixing Language Models [paper](https://arxiv.org/abs/2310.12963)
    * route queries to LLMs based on the correctness of smaller language models

## Alignment
* SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF [paper](https://arxiv.org/abs/2310.05344), [model](https://huggingface.co/nvidia/SteerLM-llama2-13B)
    *  aligning LLMs without using RLHF

## Datasets
* Proof-Pile-2 dataset: the dataset involves scientific paper, web data containing mathematics, and mathematical code. [link](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
    * #llemma
    * AlgebraicStack includes many coding language data. There are 954.1 M tokens for C++ code.
* RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset [code](https://github.com/togethercomputer/RedPajama-Data)


## Domain adaptation
* LLMs for Chip Design [paper](https://arxiv.org/abs/2311.00176) (from NVIDIA)


## Review paper
* Large Language Models for Software Engineering: Survey and Open Problems [paper](https://arxiv.org/abs/2310.03533)
    * LLM for code generation
    * LLM for software testing, debugging, repair
    * LLM for documentation generation
* Software testing with large language model: Survey, landscape, and vision [paper](https://arxiv.org/abs/2307.07221)
* Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across **Computer Vision** Tasks [paper](https://arxiv.org/pdf/2310.19909.pdf) (NeurIPS'23)
  
