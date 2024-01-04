# LLM-paper-list
There are many papers and blogs related to LLM emerge every week. So I created this list to collect papers that I'm interested in.
This list was created on Oct. 23, 2023, so some important papers before this date might be missed.

## Models
* Llemma: An Open Language Model For Mathematics [paper](https://arxiv.org/abs/2310.10631), [code](https://github.com/EleutherAI/math-lm)
    * #mathematics #codellama
 
* Magicoder: Source Code Is All You Need (a fully open sourced coding model) [paper](https://arxiv.org/pdf/2312.02120.pdf), [code](https://github.com/ise-uiuc/magicoder)

## Transformer design
* Simplifying Transformer Blocks [paper](https://arxiv.org/abs/2311.01906)
* Alternating Updates for Efficient Transformers (Google Research) (NeutIPS'23) [paper](https://arxiv.org/pdf/2301.13310.pdf)

## Multimodal LLM
* CogVLM: Visual Expert for Pretrained Language Models [paper](https://arxiv.org/abs/2311.03079), [code](https://github.com/THUDM/CogVLM)
* Emu Edit: Precise Image Editing via Recognition and Generation Tasks (from Meta) (use text instructions to modify images) [paper](https://emu-edit.metademolab.com/assets/emu_edit.pdf), [blog](https://emu-edit.metademolab.com/?utm_source=twitter&utm_medium=organic_social&utm_campaign=emu&utm_content=thread)
* LLaVA: Large Language and Vision Assistant (from Microsoft) (NeurIPS'23) [Main Page](https://llava-vl.github.io/), [code](https://github.com/haotian-liu/LLaVA)
* Mirasol3B: A Multimodal Autoregressive model for time-aligned and contextual modalities (from Google DeepMind) (text, video, audio) [paper](https://arxiv.org/pdf/2311.05698.pdf)

## Effictive training
* FP8-LM: Training FP8 Large Language Models [paper](https://arxiv.org/pdf/2310.18313.pdf), [code](https://azure.github.io/MS-AMP/)
* Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer [paper](https://arxiv.org/abs/2203.03466), [code](https://github.com/microsoft/mup) 
* MoLORA (Mixture of LORA) (from cohere) [paper](https://arxiv.org/abs/2309.05444), [code](https://github.com/for-ai/parameter-efficient-moe)
* Gated Linear Attention Transformers with Hardware-Efficient Training [paper](https://arxiv.org/pdf/2312.06635.pdf)
  * Performance still worse than transformer-based model

## Hyperparameter tuning
* Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments  (from lightning AI) [blog](https://lightning.ai/pages/community/lora-insights/)

## Parameter-Efficient fine-tuning
* LoRA: Low-Rank Adaptation of Large Language Models [paper](https://arxiv.org/pdf/2106.09685.pdf)
* LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment [paper](https://arxiv.org/abs/2312.09979v2)
  * Dealing with world knowledge forgetting during SFT

## Mixture of Experts
* A curated reading list of research in Adaptive Computation (AC) & Mixture of Experts (MoE) [repo](https://github.com/koayon/awesome-adaptive-computation)
* MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (“Here's the paper you need to read understand today” - Sasha Rush) [paper](https://arxiv.org/pdf/2211.15841.pdf) 
* Mistral MoE base model [blog](https://mistral.ai/news/mixtral-of-experts/)
* Calculate an MoE model by hand [post](https://www.linkedin.com/posts/tom-yeh_deeplearning-generatieveai-llms-activity-7141461533112381441-J35v?utm_source=share&utm_medium=member_desktop)
## After tuning
* Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch [paper](https://huggingface.co/papers/2311.03099), [code](https://github.com/yule-BUAA/MergeLM)
  - Merge SFT models into base LLM with special method can improve the performance.

## Efficient inference
* s-lora (batch lora weight inferencing) [code](https://github.com/S-LoRA/S-LoRA)
* blogs:
   * [LLM系列笔记：LLM Inference量化分析与加速](https://zhuanlan.zhihu.com/p/642272677)
   * How to make LLMs go fast [blog](https://vgel.me/posts/faster-inference/)
* PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU [paper](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf), [code](https://github.com/SJTU-IPADS/PowerInfer)
     * significantly reduces GPU memory demands and CPU-GPU data transfer
* LLM in a flash: Efficient Large Language Model Inference with Limited Memory (From Apple) [paper](https://arxiv.org/abs/2312.11514)

### Making decoding process faster:
* Lookahead Decoding [blog](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
* PaSS: Parallel Speculative Sampling (from Apple)(NeurIPS'23) [paper](https://arxiv.org/pdf/2311.13581.pdf)

## In-context learning
* Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models (from DeepMind) [paper](https://arxiv.org/abs/2311.00871)


## Long-sequence
* Efficient Streaming Language Models with Attention Sinks [paper](https://arxiv.org/abs/2309.17453), [open source implementation](https://github.com/tomaarsen/attention_sinks)
* YaRN: Efficient Context Window Extension of Large Language Models [paper](https://arxiv.org/abs/2309.00071), [YaRN on Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)
* RoPE scaling [post](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/), [Hugging Face implementation](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig.rope_scaling)
* Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey [paper](https://arxiv.org/abs/2311.12351), [repo](https://github.com/Strivin0311/long-llms-learning)

## Dynamic Adaptive Prompt Engineering
* Chain of Code:Reasoning with a Language Model-Augmented Code Emulator (from DeepMind and Fei Fei Li) [project page](https://sites.google.com/view/chain-of-code), [paper](https://arxiv.org/abs/2312.04474?utm_source=alphasignalai.beehiiv.com&utm_medium=referral&utm_campaign=chain-of-code-is-here)

## Evaluation
* PromptBench - a unified library that supports comprehensive evaluation and analysis of LLMs [code](https://github.com/microsoft/promptbench)

## RAG
* Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection [paper](https://arxiv.org/abs/2310.11511), [main page](https://selfrag.github.io/)
    *  learns to retrieve, generate and critique to enhance LM's output quality and factuality, outperforming ChatGPT and retrieval-augmented LLama2 Chat on six tasks.
* Understanding Retrieval Augmentation for Long-Form Question Answering [paper](https://arxiv.org/abs/2310.12150)
    * evidence documents should be carefully added to the LLM
    *  the order of information presented in evidence documents will impact the order of information presented in the generated answer
* Learning to Filter Context for Retrieval-Augmented Generation [paper](https://arxiv.org/abs/2311.07989), [code](https://github.com/zorazrw/filco)
* Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models [paper](https://arxiv.org/abs/2311.09210)
* Retrieval-Augmented Generation for Large Language Models: A Survey [paper](https://arxiv.org/pdf/2312.10997.pdf)

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
* Open Platypus [paper](https://arxiv.org/abs/2308.07317), [data](https://huggingface.co/datasets/garage-bAInd/Open-Platypus), data size: 24,926
* Generative AI for Math: Part I - MathPile: A Billion-Token-Scale Pretraining Corpus for Math [paper](https://arxiv.org/abs/2312.17120), [code](https://github.com/GAIR-NLP/MathPile/), [HF dataset page](https://huggingface.co/datasets/GAIR/MathPile)
    * tokenizer: GPTNeoX-20B
    * 
    
## Benchmarks
* GPQA: A Graduate-Level Google-Proof Q&A Benchmark (very hard questions) (from Cohere, Anthropic, NYU) [paper](https://arxiv.org/pdf/2311.12022.pdf), [data and code](https://github.com/idavidrein/gpqa/), data size: 448
* GAIA: a benchmark for General AI Assistants (from Meta, Yann LeCun) [paper](https://huggingface.co/papers/2311.12983), [page](https://huggingface.co/gaia-benchmark)

## Domain adaptation
* LLMs for Chip Design [paper](https://arxiv.org/abs/2311.00176) (from NVIDIA)

## Adversarial Attacks
* Adversarial Attacks on GPT-4 via Simple Random Search [paper](https://www.andriushchenko.me/gpt4adv.pdf)

## Review paper
* Large Language Models for Software Engineering: Survey and Open Problems [paper](https://arxiv.org/abs/2310.03533)
    * LLM for code generation
    * LLM for software testing, debugging, repair
    * LLM for documentation generation
* Software testing with large language model: Survey, landscape, and vision [paper](https://arxiv.org/abs/2307.07221)
* A Survey on Language Models for Code [paper](https://arxiv.org/abs/2311.07989), [code](https://github.com/codefuse-ai/Awesome-Code-LLM)
* Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across **Computer Vision** Tasks [paper](https://arxiv.org/pdf/2310.19909.pdf) (NeurIPS'23)
* The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 [paper](https://arxiv.org/abs/2311.07361)
* Data Management For Large Language Models: A Survey (for pretraining and SFT) [paper](https://arxiv.org/pdf/2312.01700.pdf), [code](https://github.com/ZigeW/data_management_LLM)
* If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents [paper](https://arxiv.org/abs/2401.00812)  
   * A comprehensive overview of the benefits of training LLMs with code-specific data.


## Not paper, but good discussion
* [Your settings are (probably) hurting your model - Why sampler settings matter](https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/)
* LLM course (got 10k stars) [repo](https://github.com/mlabonne/llm-course?tab=readme-ov-file)
