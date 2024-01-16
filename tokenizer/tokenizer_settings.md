- PaLM
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/05255347-a91c-4dce-8f1c-8d6151f5e408)
 
- CodeParrot 設定
  - https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot
  -  
- GPT-neox (Last update: 2022/04/14)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/4c474fb8-4671-4448-9ea3-55ab675dccc5)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/9880c2cc-3f75-4382-b10d-939e6c838f46)
  - GPT neox appendix 還有 whitespace的討論，這裡就不截圖了。
  - MPT-7B 用 GPT nexo 設定

- ReplitLM
  - https://github.com/replit/ReplitLM/tree/main/replit-code-v1-3b
  - 使用 MosaicML 平台訓練的
  - "We have trained a custom SentencePiece **Unigram tokenizer** optimized with a vocabulary specifically for code of 32768 tokens."
  -  
- INCODER (Last update: 2023/04/09) (ICLR 2023)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/3994e6c3-1276-4aad-926b-7f6399cfba48)
  - 內容我沒細看... 有 infilling
  
- Starcoder (Last update: 2023/12/13)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/0da2664b-72ab-45a7-86b1-8c8d9b92cf57)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/0ed78df2-6e35-48e2-93bf-a3b51cd6723a)
- Falcon
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/2a2dbe60-6fbe-4c97-addb-c4f4aa21da25)
  - "Each data split uses a dedicated tokenizer"? 有一段提到，沒很懂

- CodeT5/CodeT5+ Salesforce AI Research
- CodeGen (ICLR 2023) (Salesforce AI Research)
  - 沒提到 tokenizer = =
  - 有一些 perplexity 計算的說明
- CodeLlama
  - infilling 相關設定
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/d193b886-2648-4e7e-8948-0cb3fe969065)
- CAT-LM (CMU) Code & Test
  - "prepare the training data, comprising of the code-test file pairs, paired with a unique token (<|codetestpair|>), as well as unpaired code and test
files. We tokenize the files using a custom-trained sentencepiece tokenizer"
  - 剩下有獎跟沒講一樣 "We first train a subword tokenizer [28] using the SentencePiece [18] toolkit with a vocabulary size of 64K tokens"
- CodeGeeX (KDD’23) (Huawei)
  - GPT-2 tokenizer based
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/31cfb321-5345-467a-9e0b-f77507a3edf1)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/8198c1bc-c970-46bd-8321-027c62525c1d)
  - ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/31c05cb7-ce2c-4ef7-92fb-6c2f4f8cf2c2)
  - "CodeGeeX2 是基于 ChatGLM2 架构加入代码预训练实现" --> 跟 CodeLlama 套路類似 continuous pretraining.
  - 有 CodeGeeX2 但跟1 差在哪沒文獻
- Polyglot-Ko: Open-Source Large-Scale Korean Language Models (Last update: 2023/06/06)
  -   其實應該看日文的... 日文跟中文比較像
  -   有先做 morphological analysis tool (pre-tokenizer)
  -   ![image](https://github.com/annahung31/LLM-paper-list/assets/7519404/aac10289-1dec-4f39-bd4c-1a5cc287f0a8)

- Training and Evaluation of a Multilingual Tokenizer for GPT-SW3 (路人 report --> 瑞典機構 https://www.ai.se/en) 
  - 完整入門介紹，但結論有沒有營養不知道
  - https://arxiv.org/pdf/2304.14780.pdf
- https://github.com/salesforce/CodeTF#ast-parser-in-multiple-languages
  - Saleforce 一站式服務 @Tony 還有東西可以挖 i.e. AST tree parser https://tree-sitter.github.io/tree-sitter/creating-parsers
  - 
