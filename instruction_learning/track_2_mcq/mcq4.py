import ollama
import pandas as pd
from tqdm import tqdm
import random
import re
import time

# === 配置 ===
INPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\track_2_mcq_input.tsv"
OUTPUT_FILE = "D:\\Pycharm\\SemEval2026\\track_2_mcq\\predictions4.csv"

# === 模型名称 ===
MODEL_NAME = "llama3:latest"

# === 加载数据 ===
df = pd.read_csv(INPUT_FILE, header=0, delimiter="\t", quoting=3)
df[['lang_region', 'q_id']] = df['id'].str.rsplit('_', n=1, expand=True)

# === 随机生成伪标签 ===
def random_mcq_answer():
    choices = ['A', 'B', 'C', 'D']
    k = random.randint(1, 4)  # 随机选1或4个选项
    return ' '.join(sorted(random.sample(choices, k)))

df['pseudo_label'] = [random_mcq_answer() for _ in range(len(df))]

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
prompt_format = """### Instruction:You are a knowledgeable local resident of {region_name}. 
Your task is to answer the following multiple-choice question based on local facts and common knowledge in {region_name}.

Below are some examples from the same region:

{demos}

Now, answer this question. Respond ONLY with the letter(s) of the correct choice(s), separated by spaces (e.g., "A", "B D", "A C D").

### Question:  {question}

A) {optA}
B) {optB}
C) {optC}
D) {optD}

### Response:"""

# === 构建demonstrations ===
def build_demos(df, exclude_idx, k=3):
    other_indices = df.index[df.index != exclude_idx].tolist()
    if len(other_indices) == 0:
        return ""
    demo_indices = random.sample(other_indices, min(k, len(other_indices)))

    demo_texts = []
    for i in demo_indices:
        q = df.loc[i, 'question']
        a, b, c, d = df.loc[i, 'option_A'], df.loc[i, 'option_B'], df.loc[i, 'option_C'], df.loc[i, 'option_D']
        ans = df.loc[i, 'pseudo_label']
        demo = f"""Question: {q}
                A) {a}
                B) {b}
                C) {c}
                D) {d}
                Response: {ans}"""
        demo_texts.append(demo)

    return "\n\n---\n\n".join(demo_texts)

# === 推理 ===
def icl_mcq_predict(row, demos: str, model: str = MODEL_NAME) -> str:
    info = LANG_RE_INFO.get(row["lang_region"], {"lang_name": "Unknown", "region_name": "Unknown Region"})
    prompt = prompt_format.format(
        region_name=info["region_name"],
        demos=demos,
        question=row['question'],
        optA=row['option_A'],
        optB=row['option_B'],
        optC=row['option_C'],
        optD=row['option_D']
    )
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        raw = response['response'].strip().upper()
        # 提取 A/B/C/D 字母，去重并排序
        selected = sorted(set(re.findall(r'[ABCD]', raw)))
        return ' '.join(selected) if selected else random.choice(['A','B','C','D'])
    except Exception as e:
        print(f"[ICL Error] {e}")
        return random.choice(['A','B','C','D'])

K = 3  # 上下文示例数量

for idx in tqdm(df.index, total=len(df), desc="ICL Inference"):
    demos = build_demos(df, exclude_idx=idx, k=K)
    pred = icl_mcq_predict(df.loc[idx], demos)
    df.loc[idx, 'pseudo_label'] = pred  # 更新伪标签

# === 保存结果 ===
result_df = df[['id', 'pseudo_label']].rename(columns={'pseudo_label': 'prediction'})
result_df.to_csv(OUTPUT_FILE, index=False)

# === 示例输出 ===
print("\nSample predictions:")
print(result_df.head())