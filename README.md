# SemEval2026-Task7-Everyday-Knowledge-Across-Diverse-Languages-and-Cultures
## Data
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/e4a5f4b9-d4ee-48e3-99c2-714d9b255d62" />

## Track1:SAQ
 包含 26 种语言，30对语言与地区，对于以特定语言提出的问题，会用该语言回答
 
| Code     | Language (Region)                | Code     | Language (Region)                | Code     | Language (Region)                | Code     | Language (Region)                |
|----------|----------------------------------|----------|----------------------------------|----------|----------------------------------|----------|----------------------------------|
| ar-DZ    | Arabic (Algeria)                 | am-ET    | Amharic (Ethiopia)               | ha-NG    | Hausa (Northern Nigeria)         | as-AS    | Assamese (Assam, India)          |
| az-AZ    | Azerbaijani (Azerbaijan)         | zh-CN    | Chinese (China)                  | id-ID    | Indonesian (Indonesia)           | su-JB    | Sundanese (West Java, Indonesia) |
| fa-IR    | Persian/Farsi (Iran)             | ko-KP    | Korean (North Korea)             | ko-KR    | Korean (South Korea)             | el-GR    | Greek (Greece)                   |
| en-GB    | English (United Kingdom)         | en-US    | English (United States)          | es-ES    | Spanish (Spain)                  | es-MX    | Spanish (Mexico)                 |
| ar-EG    | Arabic (Egypt)                   | ar-MA    | Arabic (Morocco)                 | ar-SA    | Arabic (Saudi Arabia)            | ja-JP    | Japanese (Japan)                 |
|    /     | Thai (Thailand)                  |    /     | Bengali (India)                  | tl-PH    | Tagalog (Philippines)            | ta-LK    | Tamil (Sri Lanka)                |
| ta-SG    | Tamil (Singapore)                | ms-SG    | Malay (Singapore)                | zh-SG    | Singaporean Mandarin (Singapore) |     /    | Taiwanese Mandarin (Taiwan)      |
| en-AU    | English (Australia)              | es-EC    | Spanish (Ecuador)                | eu-ES    | Basque (Basque Country, Spain)   | bg-BG    | Bulgarian (Bulgaria)             |
| fr-FR    | French (France)                  | ga-IE    | Irish (Ireland)                  | sv-SE    | Swedish (Sweden)                 | cy-GB    | Welsh (Wales, UK)                |

 ## Instruction Learning
 ### google/gemma-3n-E4B-it
 - 部分回答
<img width="450" height="800" alt="image" src="https://github.com/user-attachments/assets/a2c4fdfb-a794-430c-949f-1a95e8057138" />

- 修改prompt后，要求回答精炼：
<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/2ffca717-a961-4059-a458-fa01f294e1c2" />


 ### llama3
- 先翻译为English，之后调用llama3模型处理，最后再翻译为指定语言
- **翻译过程**：选择采用Qwen-MT-turbo进行翻译，可以在[官网](https://bailian.console.aliyun.com)进行查看API调用方法，具体代码可以参考translate_touyi.py
- **模型调用过程**：选择采用Ollama调用本地部署模型，同时略修改了prompt，使其生成回答更加简练
- <img width="500" height="39" alt="image" src="https://github.com/user-attachments/assets/3eaa888e-571d-4eda-b3c0-0308d266b1cb" />
```
    prompt = f"""### Instruction: You are a local resident of {country_name}. Answer the following question in English, concisely and with cultural accuracy. Provide only the essential answer without any explanation, introduction, or punctuation.
                 ### Question: {question_en}
                 ### Response:"""
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    ans = response['message']['content'].strip()
```
- 部分回答：
  
  <img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/a90ee88a-c1a2-429a-a8a4-d6471ddd9e4e" />

  具体结果可以查看instruction_learning/track_1_saq/answers.tsv
- 其中"ga"语言的回答未能正确生成，由于Qwen不支持‘ga’。除了Qwen-MT的API，可以尝试DeepL、Google等翻译器的API，由于笔者没有国外账户，无法使用，同时若可调用baidu企业至尊版通用翻译API也可完成翻译。
- 之后可尝试直接调用大模型设计指令完成翻译，通过试验发现，Qwen-MT模型其实是能够完成翻译的（可实现翻译为英文），但由英文进行反向翻译时出现报错，或许可通过手动设置指令指示目标语言完成翻译。

## Instrcution_tuning
### 生成训练数据 见build_data.ipynb
主要采取指令学习的方法，对于每个地区，固定询问问题，生成演示数据，演示数据中的问题为英文，共计1973条数据。
观察生成数据可以发现，生成数据的质量较差，可能是由于选定生成数据答案的模型不太了解部分地区的日常知识，比如“What is the emergency telephone number for police in Algeria?”与“What number should I call for an ambulance in Algeria?”的回答相同，均为“17”。

### 下游微调——简答题
采用Lora微调，基础模型为Qwen3-4B，执行代码见track_1_instruction_tuning.ipynb
模型部分结果展示：

<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/6ac399b3-a1a5-43bf-ba48-84a0293e48cc" />




---
## Track2:MCQ
1. 将option与question拼接进行判断：
   - 翻译为轴枢语言:
    见代码process_questions1.py及mcq1.py
   - 直接拼接：
    见代码process_questions2.py
 2. 将question与4个option一起加入prompt，让模型直接回答：
     见代码process_questions3.py
 3. 上下文学习：
    首先随机生成伪标签，之后将若干条demonstrations（question and label）作为指令上下文指示模型回答，之后根据回答更新标签。
    见代码process_questions4.py及mcq4.py
### 结果：
- 拼接question与option：
  1. 使用轴枢语言：
     
     <img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/c040aea0-fcd8-413e-bae6-d66e59d099b5" />

  2. 不使用轴枢语言：
     
     <img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/4bc3c50e-83cc-4d34-a035-4902d52da7cd" />

- 直接进行多选指令学习：
  
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/743fc693-b3ce-44aa-9ab4-3ead1a3e7df2" />

- 上下文学习：
  
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/8ef0a9b0-d775-42bc-9291-88a2cf572234" />

## Instrcution_tuning
### 生成训练数据 见build_data.ipynb
同Track1
在训练过程中将生成数据问题与答案进行拼接，正确答案，answer设置为yes，错误答案设置为no
### 下游微调——多选题
采用Lora微调，基础模型为Qwen3-4B，执行代码见track_2_instruction_tuning.ipynb

模型部分结果展示：

<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/29221226-4e85-4daf-97c4-b775eb64c3db" />


## FacebookAI/xlm-roberta-large
由于 xlm-roberta-large 模型不是生成式模型，但支持 "fill-mask" pipeline, 故通过修改问题模板，将需要回答的部分作为 mask, 进而利用模型生成回答，最后将问题与回答保存在文件中。

根据生成的数据利用 Qwen-8B 进行指令微调：
- track1：
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/7e5344aa-edaa-4ac1-961e-934b5cf9cd7f" />

- track2：

<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/769e1b45-f38e-48de-b405-b9c8c1a9c8e5" />

## 训练数据 BLEnd
由于BLEnd
- track1
  
- track2

 
  




  
