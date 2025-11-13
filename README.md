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

 ### Instruction Learning
 ## google/gemma-3n-E4B-it
 - 部分回答
<img width="450" height="800" alt="image" src="https://github.com/user-attachments/assets/a2c4fdfb-a794-430c-949f-1a95e8057138" />

- 修改prompt后，要求回答精炼：
<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/2ffca717-a961-4059-a458-fa01f294e1c2" />


 ## llama3
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

---
## Track2:MCQ
1. 思路：将答案拼接到问题上作为模型输入，尝试了将问题翻译为英文和未翻译为英语两个版本，接着让模型进行判断是否匹配
2. 结果：出现答案不一致的问题，无法正确处理


 
  




  
