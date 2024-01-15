
### What kind of tasks are this method suitable for?
* reasoning problems
* coding


### Method
* 圖說： red: 可被執行的 code;  Purple: 被 LM 執行的 code
  interesting example:
<img width="394" alt="Screen Shot 2024-01-16 at 12 00 58 AM" src="https://github.com/annahung31/LLM-paper-list/assets/39179888/16027542-4337-488f-a05b-0e32d2c08035">


### Evaluation<img width="765" alt="Screen Shot 2024-01-16 at 12 03 30 AM" src="https://github.com/annahung31/LLM-paper-list/assets/39179888/2d7af88e-76d5-4f5f-ba51-c386aeaad8aa">

* benchmarks: BIG-Bench Hard
* CoC (python): 用 python 執行整段 code, if code is not executable -> marked as failure. (為了跟 program of thoughts 做對照）可預期這個方法在某些 benchmark 上會很爛，因為要寫出完全可執行的 code 幾乎不可能（例如：writing code to judge if a phrase is sarcastic.）。所以這個方法可以只看純演算法類的 benchmark 比較公平。
* CoC (Ours) = CoC (Interweave), 是這篇 paper 提出的主要方法
* 

  
### Ablation Studies 
為了檢測是否每一個部分都是有用的，設計了以下變形：
* CoC (try Python except LM)


  <img width="797" alt="Screen Shot 2024-01-16 at 12 05 53 AM" src="https://github.com/annahung31/LLM-paper-list/assets/39179888/49e06051-9641-4f75-be94-c287b39cd6af">
