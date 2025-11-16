import pandas as pd
from tqdm import tqdm
import time

from translate_qwen import translate_to_english

INPUT_FILE = r"D:\Pycharm\SemEval2026\track_2_mcq\track_2_mcq_input.tsv"
OUTPUT_FILE = r"D:\Pycharm\SemEval2026\track_2_mcq\ex_questions1.csv"
DELAY = 0.5

df = pd.read_csv(INPUT_FILE, delimiter="\t", quoting=3)
df[['lang_region', 'q_id']] = df['id'].str.rsplit('_', n=1, expand=True)

expanded = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Building & Translating"):
    q_id = row["id"]
    lang_region = row["lang_region"]
    orig_question = row["question"]

    for opt in ["A", "B", "C", "D"]:
        orig_option = row[f"option_{opt}"]

        if orig_question.endswith("?"):
            statement_orig = f"The answer to '{orig_question}' is: {orig_option}."
        else:
            statement_orig = f"{orig_question} {orig_option}"

        # 翻译成英文
        input_text_en = translate_to_english(statement_orig, lang_region)

        expanded.append({
            "id": q_id,
            "lang_region": lang_region,
            "option_letter": opt,
            "input_text": input_text_en or "[TRANSLATION_FAILED]",
        })

    time.sleep(DELAY)

expanded_df = pd.DataFrame(expanded)
expanded_df.to_csv(OUTPUT_FILE, sep="\t", index=False, quoting=3)