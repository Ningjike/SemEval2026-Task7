import sys
import os
import requests

import pandas as pd
from tqdm import tqdm
import time
from translate_qwen import translate_answer_back
from openai import OpenAI, APIError
"""
将多个模型联合起来进行指令学习
"""

# 模型API配置
INPUT_FILE = "D:\\Pycharm\\SemEval2026\\sem2\\track1\\track_1_saq_input_en.tsv"
DELAY = 0.8
MAX_RETRIES = 3  # API调用重试次数
MAX_TOKENS = 1024

# 模型API配置
MODEL_API_CONFIGS = {
    'lamma': {
        'type': 'ollama',  # 标识为ollama模型
        'base_url': 'http://localhost:11434',  # ollama默认本地地址
        'model': 'llama3:latest',  # ollama本地部署的模型名称
        'temperature': 0.1,
    },
    'qwen': {
        'api_key': 'xxxxx',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-plus',
        'temperature': 1.1,
    },
    'deepseek': {
        'api_key': 'xxxxx',
        'base_url': 'https://api.deepseek.com',
        'model': 'deepseek-chat',
        'temperature': 0.1,
    },
    'gpt-5': {
        'api_key': 'xxxxx',
        'base_url': 'https://xxxxx',
        'model': 'gpt-5-mini',
        'temperature': 0.1,
    },
    'gemini': {
        'api_key': 'xxxxx',
        'base_url': 'https://xxxxx',
        'model': 'gemini-3-flash-preview',
        'temperature': 0.1,
    },
}

# 国家代码到国家名称的映射
COUNTRY_NAME_MAP = {
    'EG': 'Egypt',
    'MA': 'Morocco',
    'SA': 'Saudi Arabia',
    'BG': 'Bulgaria',
    'GR': 'Greece',
    'AU': 'Australia',
    'GB': 'United Kingdom',
    'EC': 'Ecuador',
    'ES': 'Spain',
    'MX': 'Mexico',
    'IR': 'Iran',
    'FR': 'France',
    'IE': 'Ireland',
    'ID': 'Indonesia',
    'JP': 'Japan',
    'KR': 'South Korea',
    'SG': 'Singapore',
    'LK': 'Sri Lanka',
    'PH': 'Philippines',
    'CN': 'China',
    'ET': 'Ethiopia',
    'DZ': 'Algeria',
    'AS': 'India (Assam)',
    'AZ': 'Azerbaijan',
    'US': 'United States',
    'PV': 'Basque Country (Spain)',
    'NG': 'Nigeria',
    'KP': 'North Korea',
    'JB': 'Indonesia (West Java)',
    'SE': 'Sweden',
    'TW': 'Taiwan',
}

# 初始化OpenAI客户端
clients = {}


def initialize_clients():
    """初始化所有模型的API客户端"""
    for model_name, config in MODEL_API_CONFIGS.items():
        if config.get('type') == 'ollama':
            continue

        clients[model_name] = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )

# 国家代码到最佳模型的映射
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


def generate_prompt(question_en: str, lang_region: str) -> str:
    """
    根据问题和语言地区生成提示词
    """
    # 从lang_region中提取国家代码（例如从"en-US"中提取"US"）
    country_code = lang_region.split('-')[-1] if '-' in lang_region else lang_region
    # 使用国家代码获取国家名称
    country_name = COUNTRY_NAME_MAP.get(country_code, "the relevant country")

    prompt = f"""### Instruction: You are a local resident of {country_name}. Answer the following question in English, concisely and with cultural accuracy. Provide only the essential answer without any explanation, introduction, or punctuation.
                    ### Question: {question_en}
                    ### Response:"""
    return prompt


def call_api(model_name: str, prompt: str) -> str:
    """
    调用API获取模型响应
    """
    config = MODEL_API_CONFIGS[model_name]

    for attempt in range(MAX_RETRIES):
        try:
            if config.get('type') == 'ollama':
                # Ollama本地模型调用
                ollama_url = f"{config['base_url']}/api/generate"
                payload = {
                    "model": config['model'],
                    "prompt": prompt,
                    "temperature": config['temperature'],
                    "max_tokens": MAX_TOKENS,
                    "stream": False
                }

                # 调用Ollama API
                response = requests.post(ollama_url, json=payload)
                response.raise_for_status()  # 检查请求是否成功

                # 解析响应
                result = response.json()
                ans = result.get('response', '').strip()
            else:
                # OpenAI兼容的API调用
                client = clients[model_name]
                messages = [{"role": "user", "content": prompt}]

                # 调用API
                completion = client.chat.completions.create(
                    model=config['model'],
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=config['temperature'],
                    timeout=15
                )

                # 提取回答
                ans = completion.choices[0].message.content.strip()

            # 处理回答
            ans = ans.replace('\n', ' ').replace('\t', ' ').strip('.,;:!?"\'')
            return ans

        except APIError as e:
            error_message = str(e)
            print(f"API Error ({model_name}, attempt {attempt + 1}/{MAX_RETRIES}): {error_message}")

        except requests.RequestException as e:
            error_message = str(e)
            print(f"Request Error ({model_name}, attempt {attempt + 1}/{MAX_RETRIES}): {error_message}")

        except Exception as e:
            error_message = str(e)
            print(f"Unexpected Error ({model_name}, attempt {attempt + 1}/{MAX_RETRIES}): {error_message}")

        # 重试机制
        if attempt < MAX_RETRIES - 1:
            wait_time = (attempt + 1) * 2  # 指数退避
            print(f"Retrying after {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Failed after {MAX_RETRIES} attempts")
            return f"[ERROR] {error_message}"


def get_answer_from_api(question_en: str, lang_region: str, model: str = None) -> str:
    """
    使用大模型API获取答案
    """
    # 根据lang_region选择最佳模型
    if model is None:
        # 从语言区域代码中提取国家代码（如 'ar-EG' -> 'EG'）
        country_code = lang_region.split('-')[1] if '-' in lang_region else lang_region
        # 使用国家代码选择最佳模型
        model = COUNTRY_CODE_BEST_MODEL.get(country_code, 'gemini') 
    # 生成提示词
    prompt = generate_prompt(question_en, lang_region)

    # 调用API
    return call_api(model, prompt)


def main():
    # 初始化API客户端
    initialize_clients()

    # 输出文件路径
    OUTPUT_FILE = "track_1_saq_prediction.tsv"
    TEMP_FILE = "track_1_saq_prediction_temp.tsv"

    # 读取输入文件
    df = pd.read_csv(INPUT_FILE, delimiter="\t", quoting=3)
    # 初始化列
    if "answer_en" not in df.columns:
        df["answer_en"] = ""
    if "prediction" not in df.columns:
        df["prediction"] = ""

    # 检查是否存在已保存的结果文件
    start_idx = 0
    if os.path.exists(TEMP_FILE):
        # 读取已保存的结果
        temp_df = pd.read_csv(TEMP_FILE, delimiter="\t", quoting=3)
        print(f"读取到临时文件，共 {len(temp_df)} 行数据")
        
        # 合并已保存的结果
        merged_count = 0
        for idx, row in temp_df.iterrows():
            if row["id"] in df["id"].values:
                df_idx = df[df["id"] == row["id"]].index[0]
                if pd.notna(row.get("answer_en")):
                    df.at[df_idx, "answer_en"] = row["answer_en"]
                if pd.notna(row.get("prediction")):
                    df.at[df_idx, "prediction"] = row["prediction"]
                merged_count += 1
        
        # 找到第一个空的prediction行作为起始点，要求当前行和下一行都为空
        found_empty = False
        for idx in df.index:
            # 检查当前行
            current_empty = pd.isna(df.at[idx, "prediction"]) or df.at[idx, "prediction"] == ""
            # 检查下一行（如果存在）
            next_empty = True
            if idx + 1 < len(df):
                next_empty = pd.isna(df.at[idx + 1, "prediction"]) or df.at[idx + 1, "prediction"] == ""
            
            # 当当前行和下一行都为空时，设为起始点
            if current_empty and next_empty:
                start_idx = idx
                found_empty = True
                break
        
        # 如果没有找到符合条件的起始点，检查是否所有行都已处理
        if not found_empty:
            # 检查是否所有行都已处理
            all_processed = True
            for idx in df.index:
                if pd.isna(df.at[idx, "prediction"]) or df.at[idx, "prediction"] == "":
                    all_processed = False
                    # 如果只有单行未处理，设为起始点
                    start_idx = idx
                    found_empty = True
                    break
            
            if all_processed:
                print("所有行都已处理完成！")
                return
        
        print(f"已找到部分结果，从第 {start_idx} 行开始处理...")
    else:
        print(f"开始处理 {len(df)} 行...")

    # 处理数据
    for idx in tqdm(range(start_idx, len(df)), desc="Processing"):
        q_en = df.at[idx, "text"]
        id_value = df.at[idx, "id"]
        # 从id中提取语言区域代码 (例如 "am-ET_001" -> "am-ET")
        lang_reg = id_value.split('_')[0]
        # 获取英文答案
        ans_en = get_answer_from_api(q_en, lang_reg)
        df.at[idx, "answer_en"] = ans_en

        if ans_en.startswith("[ERROR]") or ans_en == "[SKIPPED]":
            df.at[idx, "prediction"] = ans_en
        else:
            # 回译答案到原始语言
            final_ans = translate_answer_back(ans_en, lang_reg)
            df.at[idx, "prediction"] = final_ans
        
        time.sleep(DELAY)

        # 每隔20条保存一次结果
        if (idx + 1) % 20 == 0 or (idx + 1) == len(df):
            print(f"保存进度... ({idx + 1} / {len(df)})")
            # 保存临时文件
            df.to_csv(TEMP_FILE, sep="\t", index=False)
            # 保存最终格式文件
            df[["id", "prediction"]].to_csv(OUTPUT_FILE, sep="\t", index=False)

    # 最终保存
    print("处理完成，保存最终结果...")
    df[["id", "prediction"]].to_csv(OUTPUT_FILE, sep="\t", index=False)
    # 复制临时文件到最终文件
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
    print(f"结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()