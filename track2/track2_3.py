import json
import os
import torch
from tqdm import tqdm
import random

from datasets import Dataset
from datasets import load_dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import pandas as pd

# 模型配置
model_id = "/root/autodl-fs/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# 国家名称映射
COUNTRY_NAME_MAP = {
    'ar-EG': 'Egypt',
    'ar-MA': 'Morocco',
    'ar-SA': 'Saudi Arabia',
    'bg-BG': 'Bulgaria',
    'el-GR': 'Greece',
    'en-AU': 'Australia',
    'en-GB': 'United Kingdom',
    'es-EC': 'Ecuador',
    'es-ES': 'Spain',
    'es-MX': 'Mexico',
    'eu-ES': 'Spain (Basque)',
    'fa-IR': 'Iran',
    'fr-FR': 'France',
    'ga-IE': 'Ireland',
    'id-ID': 'Indonesia',
    'ja-JP': 'Japan',
    'ko-KR': 'South Korea',
    'ms-SG': 'Singapore (Malay)',
    'ta-LK': 'Sri Lanka (Tamil)',
    'ta-SG': 'Singapore (Tamil)',
    'tl-PH': 'Philippines (Tagalog)',
    'zh-CN': 'China',
    'zh-SG': 'Singapore (Chinese)',
    'am-ET': 'Ethiopia',
    'ar-DZ': 'Algeria',
    'as-AS': 'India (Assam)',
    'az-AZ': 'Azerbaijan',
    'en-SG': 'Singapore',
    'en-US': 'United States',
    'eu-PV': 'Basque Country (Spain)',
    'ha-NG': 'Nigeria',
    'ko-KP': 'North Korea',
    'su-JB': 'Indonesia (West Java)',
    'sv-SE': 'Sweden',
    'zh-TW': 'Taiwan',
    'en-AS': 'India (Assam)',
    'en-AZ': 'Azerbaijan',
    'en-BG': 'Bulgaria',
    'en-CN': 'China',
    'en-DZ': 'Algeria',
    'en-EC': 'Ecuador',
    'en-EG': 'Egypt',
    'en-ES': 'Spain',
    'en-ET': 'Ethiopia',
    'en-FR': 'France',
    'en-GR': 'Greece',
    'en-ID': 'Indonesia',
    'en-IE': 'Ireland',
    'en-IR': 'Iran',
    'en-JB': 'Indonesia (West Java)',
    'en-JP': 'Japan',
    'en-KP': 'North Korea',
    'en-KR': 'South Korea',
    'en-LK': 'Sri Lanka',
    'en-MA': 'Morocco',
    'en-MX': 'Mexico',
    'en-NG': 'Nigeria',
    'en-PH': 'Philippines',
    'en-PV': 'Basque Country (Spain)',
    'en-SA': 'Saudi Arabia',
    'en-SE': 'Sweden',
    'en-TW': 'Taiwan',
}

def load_dataset_from_tsv(tsv_file):
    """
    从新的TSV格式加载训练数据
    """
    print(f"Loading dataset from {tsv_file}...")
    df = pd.read_csv(tsv_file, delimiter="\t")
    print(f"Loaded {len(df)} raw samples.")

    instructions = []
    responses = []
    countries = []

    valid_answers = {"A", "B", "C", "D"}

    for _, row in df.iterrows():
        # 提取字段
        lang_reg = str(row["lang_reg"]).strip()
        question = str(row["question"]).strip()
        multiple_choice_options = str(row["multiple_choice_options"]).strip()
        correct_answer = str(row["correct_answer"]).strip()

        # 从lang_reg中提取国家名称
        country = COUNTRY_NAME_MAP.get(lang_reg, "the relevant country")

        # 处理选项
        options = multiple_choice_options.strip().split('\n')
        # 清理选项
        options = [opt.strip() for opt in options if opt.strip()]

        if len(options) < 4:
            continue

        # 构建选项文本
        choices_text = ""
        for i, opt in enumerate(options[:4]):
            letter = chr(65 + i)  # A, B, C, D
            choices_text += f"{letter}. {opt}\n"

        # 生成指令
        instruction = (
            f"As a local resident of {country}, please answer the following question based on common knowledge in {country}.\n"
            f"Question: {question}\n"
            f"Options:\n{choices_text}"
            f"Answer with only the letter (A, B, C, or D):"
        )

        # 确定正确答案的字母
        # 由于新格式直接给出正确答案文本，我们需要匹配到对应的选项字母
        correct_letter = ""
        for i, opt in enumerate(options[:4]):
            if correct_answer.strip() == opt.strip():
                correct_letter = chr(65 + i)
                break

        if correct_letter not in valid_answers:
            continue

        instructions.append(instruction)
        responses.append(correct_letter)
        countries.append(country)

    df = pd.DataFrame({
        "instruction": instructions,
        "response": responses,
        "country": countries
    })

    print(f"Processed {len(df)} valid samples after filtering.")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df

EOS_TOKEN = tokenizer.eos_token

# 加载数据
train_df, val_df = load_dataset_from_tsv("trial_data_multiple_choice.tsv")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses = examples["response"]
    texts = [
        f"### Instruction:\n{instr}\n\n### Response:\n{resp}{EOS_TOKEN}"
        for instr, resp in zip(instructions, responses)
    ]
    return {"text": texts}


train_dataset = Dataset.from_pandas(train_df).map(formatting_prompts_func, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(formatting_prompts_func, batched=True)

# LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
os.makedirs('./autodl-tmp/checkpoint', exist_ok=True)
training_args = TrainingArguments(
    output_dir="./autodl-tmp/checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
    optim="paged_adamw_8bit",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    tf32=False,
    seed=3407,
    report_to="none",
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
model.save_pretrained("./autodl-tmp/final_lora")
tokenizer.save_pretrained("./autodl-tmp/final_lora")