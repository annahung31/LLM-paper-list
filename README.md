# LLM-paper-list
There are many papers and blogs related to LLM emerge every week. So I created this list to collect papers that I'm interested in.
This list was created on Oct. 23, 2023, so some important papers before this date might be missed.

## Models
* Llemma: An Open Language Model For Mathematics [paper](https://arxiv.org/abs/2310.10631), [code](https://github.com/EleutherAI/math-lm)
    * #mathematics #codellama

## Transformer design
* Simplifying Transformer Blocks [paper](https://arxiv.org/abs/2311.01906)
* Alternating Updates for Efficient Transformers (Google Research) (NeutIPS'23) [paper](https://arxiv.org/pdf/2301.13310.pdf)

## bi-modal LLM
* CogVLM: Visual Expert for Pretrained Language Models [paper](https://arxiv.org/abs/2311.03079), [code](https://github.com/THUDM/CogVLM)
* Emu Edit: Precise Image Editing via Recognition and Generation Tasks (from Meta) (use text instructions to modify images) [paper](https://emu-edit.metademolab.com/assets/emu_edit.pdf), [blog](https://emu-edit.metademolab.com/?utm_source=twitter&utm_medium=organic_social&utm_campaign=emu&utm_content=thread)
* LLaVA: Large Language and Vision Assistant (from Microsoft) (NeurIPS'23) [Main Page](https://llava-vl.github.io/), [code](https://github.com/haotian-liu/LLaVA)

## Effictive training
* FP8-LM: Training FP8 Large Language Models [paper](https://arxiv.org/pdf/2310.18313.pdf), [code](https://azure.github.io/MS-AMP/)
* Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer [paper](https://arxiv.org/abs/2203.03466), [code](https://github.com/microsoft/mup) 
* MoLORA (Mixture of LORA) (from cohere) [paper](https://arxiv.org/abs/2309.05444), [code](https://github.com/for-ai/parameter-efficient-moe)

## Hyperparameter tuning
* Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments  (from lightning AI) [blog](https://lightning.ai/pages/community/lora-insights/)

## Parameter-Efficient fine-tuning
* LoRA: Low-Rank Adaptation of Large Language Models [paper](https://arxiv.org/pdf/2106.09685.pdf)

## Mixture of Experts
* A curated reading list of research in Adaptive Computation (AC) & Mixture of Experts (MoE) [repo](https://github.com/koayon/awesome-adaptive-computation)

## Efficient inference
* s-lora (batch lora weight inferencing) [code](https://github.com/S-LoRA/S-LoRA)
* blogs:
   * [LLM系列笔记：LLM Inference量化分析与加速](https://zhuanlan.zhihu.com/p/642272677)

## In-context learning
* Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models (from DeepMind) [paper](https://arxiv.org/abs/2311.00871)


## Long-sequence
* Efficient Streaming Language Models with Attention Sinks [paper](https://arxiv.org/abs/2309.17453), [open source implementation](https://github.com/tomaarsen/attention_sinks)
* YaRN: Efficient Context Window Extension of Large Language Models [paper](https://arxiv.org/abs/2309.00071), [YaRN on Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)
* RoPE scaling [post](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/), [Hugging Face implementation](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig.rope_scaling)

## Dynamic Adaptive Prompt Engineering


## RAG
* Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection [paper](https://arxiv.org/abs/2310.11511), [main page](https://selfrag.github.io/)
    *  learns to retrieve, generate and critique to enhance LM's output quality and factuality, outperforming ChatGPT and retrieval-augmented LLama2 Chat on six tasks.
* Understanding Retrieval Augmentation for Long-Form Question Answering [paper](https://arxiv.org/abs/2310.12150)
    * evidence documents should be carefully added to the LLM
    *  the order of information presented in evidence documents will impact the order of information presented in the generated answer
* Learning to Filter Context for Retrieval-Augmented Generation [paper](https://arxiv.org/abs/2311.07989), [code](https://github.com/zorazrw/filco)
* Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models [paper](https://arxiv.org/abs/2311.09210)

## Cost saving
* AutoMix: Automatically Mixing Language Models [paper](https://arxiv.org/abs/2310.12963)
    * route queries to LLMs based on the correctness of smaller language models

## Alignment
* SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF [paper](https://arxiv.org/abs/2310.05344), [model](https://huggingface.co/nvidia/SteerLM-llama2-13B)
    *  aligning LLMs without using RLHF
* Fine-tuning Language Models for Factuality [paper](https://arxiv.org/abs/2311.08401)

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
* A Survey on Language Models for Code [paper](https://arxiv.org/abs/2311.07989), [code](https://github.com/codefuse-ai/Awesome-Code-LLM)
* Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across **Computer Vision** Tasks [paper](https://arxiv.org/pdf/2310.19909.pdf) (NeurIPS'23)
* The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 [paper](https://arxiv.org/abs/2311.07361)


## Not paper, but good discussion
* [Your settings are (probably) hurting your model - Why sampler settings matter](https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/)
