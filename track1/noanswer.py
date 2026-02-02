import pandas as pd

INPUT_FILE = "D:\\Pycharm\\SemEval2026\\sem2\\track1\\track_1_saq_prediction_translated.tsv"
OUTPUT_FILE = "D:\\Pycharm\\SemEval2026\\sem2\\track_1_saq_prediction.tsv"
df = pd.read_csv(INPUT_FILE, sep="\t")

# 从id中提取国家代码-序号（例如从"am-ET_001"中提取"ET-001"）
df["country_seq"] = df["id"].apply(lambda x: x.split("-")[1] if "-" in x else x)

# 找出所有prediction为空的行对应的国家代码-序号
empty_prediction_country_seqs = df[df["prediction"].isnull() | (df["prediction"] == "")]["country_seq"].unique()

# 将这些国家代码-序号对应的所有prediction都设置为"no answer"
if len(empty_prediction_country_seqs) > 0:
    print(f"发现空prediction的国家代码-序号: {empty_prediction_country_seqs}")
    df.loc[df["country_seq"].isin(empty_prediction_country_seqs), "prediction"] = "no answer"
else:
    print("没有发现空prediction")

# 确保所有空值都被设置为"no answer"
df["prediction"] = df["prediction"].fillna("no answer")
df.loc[df["prediction"] == "", "prediction"] = "no answer"

# 删除临时列
df = df.drop(columns=["text", "country_seq"])
df.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"处理完成，结果已保存到 {OUTPUT_FILE}")
