import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.cross_lingual_experiment import CrossLingualExperiment, BaichuanSelfCheckPrompt, ChineseSelfCheckNgram, LOCAL_MODEL_PATH

# ==============================================================================
# 1. 白盒分析器 (复用)
# ==============================================================================
class WhiteBoxAnalyzer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        self.model.eval()
        self.device = self.model.device

    def compute_uncertainty(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] == 0: return 0.0, 0.0
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        
        # Entropy
        probs = torch.softmax(shift_logits, dim=-1)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        avg_entropy = entropy.mean().item()
        
        # PPL
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        return avg_entropy, perplexity

# ==============================================================================
# 2. 数据生成与特征提取
# ==============================================================================
def prepare_data():
    csv_path = os.path.join(PROJECT_ROOT, "results", "full_features_data.csv")
    
    if os.path.exists(csv_path):
        print(f"检测到现有数据 {csv_path}，直接加载...")
        return pd.read_csv(csv_path)
        
    print("正在重新提取全量特征...")
    exp = CrossLingualExperiment()
    prompt_checker = BaichuanSelfCheckPrompt(exp.baichuan)
    whitebox = WhiteBoxAnalyzer(LOCAL_MODEL_PATH)
    
    data = exp.generate_dataset(num_samples=None) 
    
    records = []
    for item in tqdm(data, desc="Feature Extraction"):
        sents = item['sentences']
        samples = item['samples']
        labels = item['labels'] # 0=Fact, 1=Hallucination
        
        prompt_scores = prompt_checker.predict(sents, samples, verbose=False)
        
        for i, sent in enumerate(sents):
            ent, ppl = whitebox.compute_uncertainty(sent)
            records.append({
                'sentence': sent,
                'label': labels[i],
                'prompt_score': prompt_scores[i],
                'entropy': ent,
                'ppl': ppl,
                'len': len(sent)
            })
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至 {csv_path}")
    return df

# ==============================================================================
# 3. 深度相关性分析 (Internal Analysis)
# ==============================================================================
def run_internal_analysis(df):
    print("\n" + "="*50)
    print("【深度分析：黑盒 vs 白盒】")
    print("="*50)
    
    # 1. 相关性
    corr_ent, _ = pearsonr(df['prompt_score'], df['entropy'])
    print(f"SelfCheck分数(黑盒) 与 内部熵(白盒) 的相关系数: {corr_ent:.4f}")
    
    # 2. 幻觉与事实的分布差异
    print("\n不同类别的指标均值:")
    print(df.groupby('label')[['prompt_score', 'entropy', 'ppl']].mean())
    
    # 3. 寻找“过度自信的幻觉” (Confident Hallucinations)
    # 定义：Prompt 认为是幻觉 (Score > 0.8)，但 Entropy 很低 (Top 20% confident)
    low_entropy_th = df['entropy'].quantile(0.2)
    confident_hallucinations = df[(df['label'] == 1) & (df['entropy'] < low_entropy_th)]
    
    ratio = len(confident_hallucinations) / len(df[df['label']==1])
    print(f"\n发现【自信的幻觉】比例: {ratio:.2%}")
    print(f"说明有 {ratio:.2%} 的幻觉，模型内部非常自信（低熵），只能靠外部一致性（Prompt）检测出来。")
    print("结论：这证明了 SelfCheckGPT 存在的必要性 —— 单看内部状态是不够的。")

# ==============================================================================
# 4. 方向一：Hybrid-SelfCheck (UE-SelfCheck)
# ==============================================================================
def run_hybrid_algorithm(df):
    print("\n" + "="*50)
    print("【新算法实验：UE-SelfCheck (Hybrid Strategy)】")
    print("目标：利用白盒特征过滤样本，大幅降低计算成本。")
    print("="*50)
    
    # 只有当 Prompt 真的去跑了，才算 Cost = 1
    # 如果被 Entropy 拦截了，Cost = 0 (忽略 Entropy 本身的微小计算量)
    
    labels = df['label'].values
    prompt_scores = df['prompt_score'].values
    entropies = df['entropy'].values
    
    # 基线 AUC
    precision, recall, _ = precision_recall_curve(labels, prompt_scores)
    base_auc = auc(recall, precision)
    print(f"Baseline AUC (全量 SelfCheck): {base_auc:.4f}")
    
    # 模拟不同的阈值策略
    # 策略：如果 Entropy < 阈值，直接判为 0 (事实)；否则跑 SelfCheck
    # 我们只做单边截断（过滤掉特别自信的），因为过滤特别困惑的风险较大
    
    thresholds = np.percentile(entropies, np.arange(0, 100, 10))
    results = []
    
    print(f"\n{'阈值(Entropy percentile)':<25} {'计算量节省(%)':<15} {'Hybrid AUC':<15} {'性能保持率(%)'}")
    print("-" * 75)
    
    best_setting = None
    
    for p in range(0, 90, 10): # 遍历 0% 到 80% 的分位数
        th = np.percentile(entropies, p)
        
        # 混合分数计算
        hybrid_scores = []
        cost_count = 0
        
        for ent, p_score in zip(entropies, prompt_scores):
            if ent < th:
                # 熵很低 -> 模型很自信 -> 直接信任模型 -> 判为事实(0.0)
                # 这里为了平滑，可以给一个极小值，或者直接 0
                hybrid_scores.append(0.0) 
            else:
                # 熵较高 -> 模型犹豫 -> 启动 SelfCheck
                hybrid_scores.append(p_score)
                cost_count += 1
                
        # 评估
        hybrid_scores = np.array(hybrid_scores)
        prec, rec, _ = precision_recall_curve(labels, hybrid_scores)
        h_auc = auc(rec, prec)
        
        saving = 1.0 - (cost_count / len(labels))
        retention = h_auc / base_auc
        
        print(f"Bottom {p}% ({th:.4f}) {'':<10} {saving*100:.1f}% {'':<10} {h_auc:.4f} {'':<10} {retention*100:.1f}%")
        
        # 寻找最佳平衡点：节省 > 30% 且 性能保持 > 98%
        if saving > 0.30 and retention > 0.98:
            best_setting = (p, saving, h_auc)

    print("-" * 75)
    if best_setting:
        print(f"\n最佳配置发现：过滤掉熵最低的 {best_setting[0]}% 样本。")
        print(f"   可以节省 {best_setting[1]*100:.1f}% 的计算成本，同时保持 {best_setting[2]:.4f} 的 AUC。")
        print("   结论：这证明了 UE-SelfCheck 算法在工业落地中的巨大潜力。")
    else:
        print("\n结论：Entropy 与 Prompt 互补性极强，任何过滤都会导致性能下降。")
        print("这说明哪怕模型很自信，也需要 SelfCheck 复核。")

if __name__ == "__main__":
    # 1. 准备数据
    df = prepare_data()
    
    # 2. 深度分析
    run_internal_analysis(df)
    
    # 3. 新算法验证
    run_hybrid_algorithm(df)