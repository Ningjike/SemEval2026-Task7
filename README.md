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
  具体结果可以查看instruction_learning/track_1_saq/answers.tsv
- 其中"ga"语言的回答未能正确生成，由于Qwen不支持‘ga’。除了Qwen-MT的API，可以尝试DeepL、Google等翻译器的API，由于笔者没有国外账户，无法使用，同时若可调用baidu企业至尊版通用翻译API也可完成翻译。
- 之后可尝试直接调用大模型设计指令完成翻译，通过试验发现，Qwen-MT模型其实是能够完成翻译的（可实现翻译为英文），但由英文进行反向翻译时出现报错，或许可通过手动设置指令指示目标语言完成翻译。

## Instrcution_tuning
### 生成训练数据 见build_data.ipynb
主要采取指令学习的方法，对于每个地区，固定询问问题，生成演示数据，演示数据中的问题为英文，共计1973条数据。
观察生成数据可以发现，生成数据的质量较差，可能是由于选定生成数据答案的模型不太了解部分地区的日常知识，比如“What is the emergency telephone number for police in Algeria?”与“What number should I call for an ambulance in Algeria?”的回答相同，均为“17”。

### 下游微调——简答题
采用Lora微调，基础模型为Qwen3-4B，执行代码见track_1_instruction_tuning.ipynb

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
## Instrcution_tuning
### 生成训练数据 见build_data.ipynb
同Track1
在训练过程中将生成数据问题与答案进行拼接，正确答案，answer设置为yes，错误答案设置为no
### 下游微调——多选题
采用Lora微调，基础模型为Qwen3-4B，执行代码见track_2_instruction_tuning.ipynb

## FacebookAI/xlm-roberta-large
由于 xlm-roberta-large 模型不是生成式模型，但支持 "fill-mask" pipeline, 故通过修改问题模板，将需要回答的部分作为 mask, 进而利用模型生成回答，最后将问题与回答保存在文件中。

## 综合模型
根据development 阶段在trail data上的各个模型进行指令学习的得分如下：
| models  | lamma                 | qwen                   | deepseek               | gpt-5                  | gemini                 | 综合                    |
| ------- | --------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | --------------------- |
| overall | 39.189189189189186    | 43.91891891891892      | 45.945945945945944     | 45.945945945945944     | 52.027027027027025     | 54.729729729729726    |
| ar-EG   | 28.571428571428573    | 28.571428571428573     | 42.857142857142854     | 42.857142857142854     | 57.142857142857146     | 57.142857142857146    |
| ar-MA   | 57.142857142857146    | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146    |
| ar-SA   | 42.857142857142854    | 42.857142857142854     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146    |
| bg-BG   | 71.42857142857143     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 85.71428571428571     |
| el-GR   | 60.0                  | 60.0                   | 60.0                   | 60.0                   | 60.0                   | 60.0                  |
| en-AU   | 28.571428571428573    | 14.285714285714286     | 28.571428571428573     | 42.857142857142854     | 42.857142857142854     | 42.857142857142854    |
| en-GB   | 60.0                  | 20.0                   | 20.0                   | 40.0                   | 40.0                   | 60.0                  |
| es-EC   | 62.5                  | 75.0                   | 87.5                   | 62.5                   | 75.0                   | 75.0                  |
| es-ES   | 60.0                  | 60.0                   | 60.0                   | 60.0                   | 60.0                   | 60.0                  |
| es-MX   | 20.0                  | 40.0                   | 20.0                   | 20.0                   | 40.0                   | 40.0                  |
| eu-ES   | 14.285714285714286    | 42.857142857142854     | 42.857142857142854     | 57.142857142857146     | 71.42857142857143      | 71.42857142857143     |
| fa-IR   | 40.0                  | 40.0                   | 40.0                   | 40.0                   | 60.0                   | 60.0                  |
| fr-FR   | 0.0                   | 0.0                    | 25.0                   | 12.5                   | 25.0                   | 12.5                  |
| ga-IE   | 14.285714285714286    | 28.571428571428573     | 0.0                    | 14.285714285714286     | 14.285714285714286     | 14.285714285714286    |
| id-ID   | 20.0                  | 40.0                   | 40.0                   | 60.0                   | 40.0                   | 60.0                  |
| ja-JP   | 14.285714285714286    | 14.285714285714286     | 14.285714285714286     | 14.285714285714286     | 14.285714285714286     | 14.285714285714286    |
| ko-KR   | 0.0                   | 0.0                    | 20.0                   | 20.0                   | 40.0                   | 20.0                  |
| ms-SG   | 57.142857142857146    | 71.42857142857143      | 57.142857142857146     | 57.142857142857146     | 71.42857142857143      | 71.42857142857143     |
| ta-LK   | 71.42857142857143     | 57.142857142857146     | 57.142857142857146     | 57.142857142857146     | 71.42857142857143      | 71.42857142857143     |
| ta-SG   | 28.571428571428573    | 28.571428571428573     | 28.571428571428573     | 14.285714285714286     | 14.285714285714286     | 28.571428571428573    |
| tl-PH   | 50.0                  | 75.0                   | 87.5                   | 75.0                   | 75.0                   | 87.5                  |
| zh-CN   | 60.0                  | 100.0                  | 80.0                   | 100.0                  | 100.0                  | 100.0                 |
| zh-SG   | 42.857142857142854    | 57.142857142857146     | 57.142857142857146     | 42.857142857142854     | 57.142857142857146     | 57.142857142857146    |

确定国家代码到最佳模型的映射：
COUNTRY_CODE_BEST_MODEL = {
    'EG': 'gemini',  # 埃及
    'MA': 'lamma',   # 摩洛哥
    'SA': 'deepseek', # 沙特阿拉伯
    'BG': 'lamma',   # 保加利亚
    'GR': 'lamma',   # 希腊
    'AU': 'gemini',   # 澳大利亚
    'GB': 'lamma',   # 英国
    'EC': 'deepseek', # 厄瓜多尔
    'ES': 'gemini',   # 西班牙
    'MX': 'qwen',    # 墨西哥
    'IR': 'gemini',  # 伊朗
    'FR': 'deepseek', # 法国
    'IE': 'qwen',    # 爱尔兰
    'ID': 'gpt-5',   # 印度尼西亚
    'JP': 'qwen',    # 日本
    'KR': 'gemini',  # 韩国
    'SG': 'qwen',    # 新加坡
    'LK': 'lamma',   # 斯里兰卡
    'PH': 'deepseek', # 菲律宾
    'CN': 'qwen',    # 中国
    'ET': 'gemini',  # 埃塞俄比亚
    'DZ': 'gemini',  # 阿尔及利亚
    'NG': 'deepseek', # 尼日利亚
    'AS': 'gpt-5',   # 印度（阿萨姆）
    'AZ': 'lamma',   # 阿塞拜疆
    'US': 'gpt-5',   # 美国
    'PV': 'gemini',  # 巴斯克地区
    'JB': 'gpt-5',   # 印度尼西亚（西爪哇）
    'SE': 'lamma',   # 瑞典
    'TW': 'qwen',    # 台湾
    'KP': 'gemini',  # 朝鲜
}

## Instruction Tuning
根据trail data 作为训练数据对Qwen-4B进行指令微调，见代码：track2\track2_3.py


  
