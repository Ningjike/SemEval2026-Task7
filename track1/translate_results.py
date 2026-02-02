#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将track1_2的回答结果按照输入文件顺序排序并翻译回原始语言
"""
import os
import pandas as pd
import time

def translate_with_google(text: str, target_lang: str) -> str:
    """
    使用Google翻译将文本翻译为目标语言
    """
    if not text:
        return text
    
    # 使用更稳定的方法，避免异步问题
    import requests
    import urllib.parse
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # 使用Google翻译的HTTP API
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "en",  # 源语言为英语
                "tl": target_lang,  # 目标语言
                "dt": "t",
                "q": text
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # 检查HTTP错误
            
            # 解析JSON响应
            data = response.json()
            translated_text = "".join([item[0] for item in data[0]])
            
            time.sleep(1)  # 控制请求频率，避免被封禁
            return translated_text
        except Exception as e:
            print(f"[Google Translate Error] 尝试 {attempt+1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 重试前等待
            else:
                return text

def main():
    # 文件路径
    input_file = "D:\\Pycharm\\SemEval2026\\sem2\\track1\\track_1_saq_input.tsv"
    prediction_file = "D:\\Pycharm\\SemEval2026\\sem2\\track1\\track_1_saq_prediction.tsv"
    output_file = "D:\\Pycharm\\SemEval2026\\sem2\\track1\\track_1_saq_prediction_translated.tsv"
    
    # 读取输入文件
    print(f"读取输入文件: {input_file}")
    input_df = pd.read_csv(input_file, delimiter="\t", quoting=3)
    print(f"输入文件共 {len(input_df)} 行")
    
    # 读取预测结果文件
    print(f"读取预测结果文件: {prediction_file}")
    prediction_df = pd.read_csv(prediction_file, delimiter="\t", quoting=3)
    print(f"预测结果文件共 {len(prediction_df)} 行")
    
    # 将预测结果转换为字典，方便根据id查找
    prediction_dict = dict(zip(prediction_df["id"], prediction_df["prediction"]))
    
    # 按照输入文件的顺序处理
    print("按照输入文件顺序处理结果...")
    results = []
    
    for idx, row in input_df.iterrows():
        id_ = row["id"]
        text = row["text"]
        
        # 查找对应的预测结果
        # 注意：预测结果中可能只有英语问题的回答，需要找到对应的英语问题id
        # 例如，对于am-ET_001，查找对应的en-ET_001的回答
        if "-" in id_ and "_" in id_:
            # 正确构造对应的英语问题ID
            # 例如：am-ET_001 -> en-ET_001
            parts = id_.split("-")
            lang = parts[0]
            rest = "-".join(parts[1:])
            en_id = f"en-{rest}"
        else:
            en_id = id_
        
        # 优先使用直接匹配的id，否则使用对应的英语id
        prediction = prediction_dict.get(id_, prediction_dict.get(en_id, ""))
        
        # 提取语言地区代码
        lang_region = id_.split("_")[0]
        
        # 提取语言代码
        lang_code = lang_region.split('-')[0].lower()
        
        # 翻译回答回原始语言
        if lang_code != "en" and prediction:
            translated_answer = translate_with_google(prediction, lang_code)
        else:
            translated_answer = prediction
        
        # 添加到结果列表
        results.append({
            "id": id_,
            "text": text,
            "prediction": translated_answer
        })
        
        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"处理进度: {idx + 1}/{len(input_df)}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 保存结果到TSV文件
    print(f"保存结果到文件: {output_file}")
    result_df.to_csv(output_file, sep="\t", index=False, quoting=3)
    print(f"保存完成，共 {len(result_df)} 行数据")

if __name__ == "__main__":
    main()
