import ollama
import pandas as pd
from tqdm import tqdm
import time
import re

# === 配置 ===
INPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\track_2_mcq_input.tsv"
OUTPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\predictions3.csv"

MODEL_NAME = "llama3:latest"

# === 加载数据 ===
df = pd.read_csv(INPUT_FILE, header=0, delimiter="\t", quoting=3)
df[['lang_region', 'q_id']] = df['id'].str.rsplit('_', n=1, expand=True)

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

# === 翻译 Prompt ===
translate_prompt_template = """Translate the following text from {lang_name} to English. Only output the translation, nothing else.
Text: {text}
"""

# === 多选题回答 Prompt ===
mcq_prompt_template = """You are an expert taking a multiple-choice quiz. Below is a question in English with four options labeled A, B, C, D.
Select all correct answers. Respond ONLY with the letters of the correct choices (e.g., "A", "B D", "A C D"). Do not explain.

Question: {question_en}

A) {choice_A_en}
B) {choice_B_en}
C) {choice_C_en}
D) {choice_D_en}

Answer:"""


def translate_text(text, lang_code):
    if lang_code.startswith("en"):
        return text
    lang_info = LANG_RE_INFO.get(lang_code, {"lang_name": "Unknown"})
    lang_name = lang_info["lang_name"]
    prompt = translate_prompt_template.format(lang_name=lang_name, text=text)
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.1})
        return response['response'].strip()
    except Exception as e:
        print(f"Translation error for '{text}': {e}")
        return text


def get_mcq_answer(question_en, choices_en):
    prompt = mcq_prompt_template.format(
        question_en=question_en,
        choice_A_en=choices_en['A'],
        choice_B_en=choices_en['B'],
        choice_C_en=choices_en['C'],
        choice_D_en=choices_en['D']
    )
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
        raw_ans = response['response'].strip().upper()
        # 提取 A/B/C/D 字母，去重并排序
        selected = sorted(set(re.findall(r'[ABCD]', raw_ans)))
        return ' '.join(selected) if selected else "A"
    except Exception as e:
        print(f"MCQ answer error: {e}")
        return "A"


# === 主循环 ===
predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    lang_region = row['lang_region']

    # Step 1: 翻译 question 和四个选项
    question_en = translate_text(row['question'], lang_region)
    choice_A_en = translate_text(row['option_A'], lang_region)
    choice_B_en = translate_text(row['option_B'], lang_region)
    choice_C_en = translate_text(row['option_C'], lang_region)
    choice_D_en = translate_text(row['option_D'], lang_region)

    # Step 2: 构造英文 MCQ 并让模型作答
    choices_en = {'A': choice_A_en, 'B': choice_B_en, 'C': choice_C_en, 'D': choice_D_en}
    pred = get_mcq_answer(question_en, choices_en)

    predictions.append({
        'id': row['id'],
        'prediction': pred
    })

    # 可选：避免 Ollama 过载（如果模型响应慢）
    time.sleep(0.1)

# === 保存结果 ===
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(OUTPUT_FILE, index=False)
