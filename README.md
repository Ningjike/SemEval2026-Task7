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
|    \     | Thai (Thailand)                  |    \     | Bengali (India)                  | tl-PH    | Tagalog (Philippines)            | ta-LK    | Tamil (Sri Lanka)                |
| ta-SG    | Tamil (Singapore)                | ms-SG    | Malay (Singapore)                | zh-SG    | Singaporean Mandarin (Singapore) |     \    | Taiwanese Mandarin (Taiwan)      |
| en-AU    | English (Australia)              | es-EC    | Spanish (Ecuador)                | eu-ES    | Basque (Basque Country, Spain)   | bg-BG    | Bulgarian (Bulgaria)             |
| fr-FR    | French (France)                  | ga-IE    | Irish (Ireland)                  | sv-SE    | Swedish (Sweden)                 | cy-GB    | Welsh (Wales, UK)                |

 ### Instruction Learning
 ## google/gemma-3n-E4B-it
 - 部分回答
<img width="450" height="800" alt="image" src="https://github.com/user-attachments/assets/a2c4fdfb-a794-430c-949f-1a95e8057138" />

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
  <img width="800" height="900" alt="image" src="https://github.com/user-attachments/assets/a90ee88a-c1a2-429a-a8a4-d6471ddd9e4e" />
  具体结果可以查看instruction_learning/track_1_saq/answers.tsv
  




  
