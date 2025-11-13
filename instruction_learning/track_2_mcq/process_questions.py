import pandas as pd
from tqdm import tqdm

import time

from translate_qwen import translate_to_english

INPUT_FILE = "D:\\Pycharm\SemEval2026\\track_2_mcq\\track_2_mcq_input.tsv"
DELAY = 0.5

df = pd.read_csv(INPUT_FILE, delimiter="\t", quoting=3)
df[['lang_region', 'q_id']] = df['id'].str.rsplit('_', n=1, expand=True)

# 初始化列
# df["question_en"] = ""
print(f"开始处理 {len(df)} 行...")

# for idx in tqdm(df.index, desc="Processing"):
#     orig_q = df.at[idx, "question"]
#     lang_reg = str(df.at[idx, "lang_region"])
#
#     # 翻译问题到英文
#     q_en = translate_to_english(orig_q, lang_reg)
#     df.at[idx, "question_en"] = q_en
#
#     if not q_en or q_en.startswith("["):
#         df.at[idx, "answer"] = "[SKIPPED] Translation failed."
#         continue
#
#     time.sleep(DELAY)


expanded = []
for _, row in df.iterrows():
    q_id = row["id"]
    question = row["question"]
    # question_en = row["question_en"]
    for opt in ["A", "B", "C", "D"]:
        option_text = row[f"option_{opt}"]
        text = f"{question}{option_text}"
        # text = f"{question_en} {option_text}"
        expanded.append({
            "id": q_id,
            "lang_region": row["lang_region"],
            "option_letter": opt,
            "input_text": text,
        })

expanded_df = pd.DataFrame(expanded)
expanded_df.to_csv("ex_questions2.csv",sep="\t", index=False, quoting=3)


