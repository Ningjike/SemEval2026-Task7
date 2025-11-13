import ollama
import pandas as pd
from tqdm import tqdm

import time

# === 配置 ===
INPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\track_2_mcq_input.tsv.csv"
OUTPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\predictions2.csv"

# === 模型名称 ===
MODEL_NAME = "llama3:latest"

# === 加载数据 ===
df = pd.read_csv(INPUT_FILE, header=0, delimiter="\t", quoting=3)

# === 区域信息映射 ===
LANG_RE_INFO = {
    # Arabic variants
    "ar-DZ": {"lang_name": "Arabic", "region_name": "Algeria"},
    "ar-EG": {"lang_name": "Arabic", "region_name": "Egypt"},
    "ar-MA": {"lang_name": "Arabic", "region_name": "Morocco"},
    "ar-SA": {"lang_name": "Arabic", "region_name": "Saudi Arabia"},

    # Amharic
    "am-ET": {"lang_name": "Amharic", "region_name": "Ethiopia"},

    # Hausa
    "ha-NG": {"lang_name": "Hausa", "region_name": "Northern Nigeria"},

    # Assamese
    "as-AS": {"lang_name": "Assamese", "region_name": "Assam, India"},

    # Azerbaijani
    "az-AZ": {"lang_name": "Azerbaijani", "region_name": "Azerbaijan"},

    # Chinese variants
    "zh-CN": {"lang_name": "Chinese", "region_name": "China"},
    "zh-SG": {"lang_name": "Singaporean Mandarin", "region_name": "Singapore"},
    "zh-TW": {"lang_name": "Taiwanese Mandarin", "region_name": "Taiwan"},

    # Indonesian
    "id-ID": {"lang_name": "Indonesian", "region_name": "Indonesia"},

    # Sundanese
    "su-JB": {"lang_name": "Sundanese", "region_name": "West Java, Indonesia"},

    # Persian/Farsi
    "fa-IR": {"lang_name": "Persian", "region_name": "Iran"},

    # Korean variants
    "ko-KP": {"lang_name": "Korean", "region_name": "North Korea"},
    "ko-KR": {"lang_name": "Korean", "region_name": "South Korea"},

    # Greek
    "el-GR": {"lang_name": "Greek", "region_name": "Greece"},

    # English variants
    "en-GB": {"lang_name": "English", "region_name": "United Kingdom"},
    "en-US": {"lang_name": "English", "region_name": "United States"},
    "en-AU": {"lang_name": "English", "region_name": "Australia"},

    # Spanish variants
    "es-ES": {"lang_name": "Spanish", "region_name": "Spain"},
    "es-MX": {"lang_name": "Spanish", "region_name": "Mexico"},
    "es-EC": {"lang_name": "Spanish", "region_name": "Ecuador"},

    # Japanese
    "ja-JP": {"lang_name": "Japanese", "region_name": "Japan"},

    # Thai
    "th-TH": {"lang_name": "Thai", "region_name": "Thailand"},

    # Bengali
    "bn-IN": {"lang_name": "Bengali", "region_name": "India"},

    # Tagalog
    "tl-PH": {"lang_name": "Tagalog", "region_name": "Philippines"},

    # Tamil variants
    "ta-LK": {"lang_name": "Tamil", "region_name": "Sri Lanka"},
    "ta-SG": {"lang_name": "Tamil", "region_name": "Singapore"},

    # Malay
    "ms-SG": {"lang_name": "Malay", "region_name": "Singapore"},

    # Basque
    "eu-ES": {"lang_name": "Basque", "region_name": "Basque Country, Spain"},

    # Bulgarian
    "bg-BG": {"lang_name": "Bulgarian", "region_name": "Bulgaria"},

    # French
    "fr-FR": {"lang_name": "French", "region_name": "France"},

    # Irish
    "ga-IE": {"lang_name": "Irish", "region_name": "Ireland"},

    # Swedish
    "sv-SE": {"lang_name": "Swedish", "region_name": "Sweden"},

    # Welsh
    "cy-GB": {"lang_name": "Welsh", "region_name": "Wales, UK"},

    # === Corresponding English entries ===
    "en-DZ": {"lang_name": "English", "region_name": "Algeria"},
    "en-ET": {"lang_name": "English", "region_name": "Ethiopia"},
    "en-NG": {"lang_name": "English", "region_name": "Northern Nigeria"},
    "en-AS": {"lang_name": "English", "region_name": "Assam, India"},
    "en-AZ": {"lang_name": "English", "region_name": "Azerbaijan"},
    "en-CN": {"lang_name": "English", "region_name": "China"},
    "en-ID": {"lang_name": "English", "region_name": "Indonesia"},
    "en-JB": {"lang_name": "English", "region_name": "West Java, Indonesia"},
    "en-IR": {"lang_name": "English", "region_name": "Iran"},
    "en-KP": {"lang_name": "English", "region_name": "North Korea"},
    "en-KR": {"lang_name": "English", "region_name": "South Korea"},
    "en-GR": {"lang_name": "English", "region_name": "Greece"},
    "en-MX": {"lang_name": "English", "region_name": "Mexico"},
    "en-ES": {"lang_name": "English", "region_name": "Spain"},
    "en-EC": {"lang_name": "English", "region_name": "Ecuador"},
    "en-PH": {"lang_name": "English", "region_name": "Philippines"},
    "en-LK": {"lang_name": "English", "region_name": "Sri Lanka"},
    "en-SG": {"lang_name": "English", "region_name": "Singapore"},
    "en-BG": {"lang_name": "English", "region_name": "Bulgaria"},
    "en-FR": {"lang_name": "English", "region_name": "France"},
    "en-IE": {"lang_name": "English", "region_name": "Ireland"},
    "en-SE": {"lang_name": "English", "region_name": "Sweden"},
    "en-WL": {"lang_name": "English", "region_name": "Wales, UK"},
}

# === Prompt 模板 ===
prompt_format = """### Instruction: You are a knowledgeable local resident of {region_name}. Your task is to verify whether the following statement — which combines a question and a proposed answer — is FACTUALLY CORRECT according to local knowledge in {region_name}.
Respond with ONLY:
- "Yes" if the statement is correct,
- "No" if it is incorrect or misleading.
Do not add any explanation, punctuation, or extra text.

### Question: {question}

### Response:
"""


def build_prompt(question: str, lang_region: str) -> str:
    info = LANG_RE_INFO.get(lang_region)
    if info is None:
        raise ValueError(f"Unsupported lang_region code: '{lang_region}'. Available: {list(LANG_RE_INFO.keys())}")
    return prompt_format.format(
        region_name=info["region_name"],
        question=question
    )


# === 推理 ===
def judge_statement(question: str, lang_region: str, model: str = MODEL_NAME) -> int:
    prompt = build_prompt(question, lang_region)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1,},
        )
        output_text = response['message']['content'].strip()

        if "Yes" in output_text:
            return 1
        elif "No" in output_text:
            return 0
        else:
            return -1

    except Exception as e:
        print(f"[Exception] Failed on input: {question[:50]}... | Error: {e}")
        return -1


# === 主流程 ===
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing") :
    q_id = row["id"]
    lang_region = row["lang_region"]
    option_letter = row["option_letter"]
    input_text = row["input_text"]

    pred = judge_statement(input_text, lang_region)
    results.append({
        "id": q_id,
        "lang_region": lang_region,
        "option_letter": option_letter,
        "input_text": input_text,
        "prediction": pred
    })

# === 保存结果 ===
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False, sep="\t")

# === 示例输出 ===
print("\nSample predictions:")
print(result_df.head())