import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
import os
import sys
# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import WikiBioDatasetLoader

# 加载Qwen-7B-Chat模型
model_name = "qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,
    device_map="auto", torch_dtype=torch.float16
).eval()

# 自评估Prompt
EVAL_PROMPT = """
Given the following biography and a specific sentence from it, determine if the sentence is factually accurate. 
Use 0 for accurate, 0.5 for minor inaccuracy, and 1.0 for major inaccuracy. Answer with only the number.

Biography: {biography}
Sentence to evaluate: {sentence}
Your answer (0/0.5/1.0):
"""

# 读取预处理数据集
data_path = "data/wikibio_processed_final.json"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_pred_scores = []
all_true_labels = []

for sample in tqdm(data, desc="SelfCheckPrompt 推理中"):
    biography = sample["gpt3_text"]
    original_sentences = sample["gpt3_sentences"]
    true_labels = sample["annotation"]
    
    for sentence, true_label in zip(original_sentences, true_labels):
        # 构建Prompt
        prompt = EVAL_PROMPT.format(biography=biography[:600], sentence=sentence)
        
        # 模型生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=10, temperature=0.0,
                top_p=1.0, do_sample=False, eos_token_id=tokenizer.eos_token_id
            )
        
        # 解析回答
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        try:
            # 提取数值
            pred_score = float([c for c in answer if c in ['0', '1', '.']][:4])
            pred_score = max(0.0, min(1.0, pred_score))
        except:
            pred_score = 0.5  # 解析失败时取默认值
        
        all_pred_scores.append(pred_score)
        all_true_labels.append(true_label)

# 计算AUC-PR
all_true_labels = np.array(all_true_labels)
all_pred_scores = np.array(all_pred_scores)
binary_true_labels = (all_true_labels > 0.0).astype(int)
precision, recall, _ = precision_recall_curve(binary_true_labels, all_pred_scores)
auc_pr = auc(recall, precision)

# 保存结果
os.makedirs("results", exist_ok=True)
results = {
    "model": "SelfCheckPrompt",
    "auc_pr": auc_pr,
    "pred_scores": all_pred_scores.tolist(),
    "true_labels": all_true_labels.tolist(),
    "dataset_path": data_path
}

with open("results/selfcheck_prompt_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSelfCheckPrompt 复现完成！")
print(f"核心指标：AUC-PR = {auc_pr:.4f}（原论文约 0.90）")
print(f"结果保存：results/selfcheck_prompt_results.json")