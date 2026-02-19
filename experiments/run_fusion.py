import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import RandomForestClassifier

# 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.cross_lingual_experiment import CrossLingualExperiment, BaichuanSelfCheckPrompt, ChineseSelfCheckNgram, LOCAL_MODEL_PATH

# ==============================================================================
# 1. 白盒分析器 (计算熵和PPL)
# ==============================================================================
class WhiteBoxAnalyzer:
    def __init__(self, model_path):
        print(f"加载模型用于白盒分析: {model_path}")
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
            
        # Shift logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        
        # Entropy
        probs = torch.softmax(shift_logits, dim=-1)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1) # [1, seq_len]
        avg_entropy = entropy.mean().item()
        
        # Perplexity (Loss)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        
        return avg_entropy, perplexity

# ==============================================================================
# 2. 主流程：特征提取与融合
# ==============================================================================
def run_fusion_experiment():
    print("L2C (Learning-to-Check) 多维特征融合")
    
    # 1. 准备数据生成器
    exp = CrossLingualExperiment()
    
    # 2. 初始化各个检测器
    # A. 黑盒 - Prompt (Standard)
    prompt_checker = BaichuanSelfCheckPrompt(exp.baichuan)
    
    # B. 黑盒 - Ngram
    ngram_checker = ChineseSelfCheckNgram(n=1)
    
    # C. 白盒 - Entropy/PPL     
    whitebox_analyzer = WhiteBoxAnalyzer(LOCAL_MODEL_PATH)

    # 3. 生成全量数据
    data = exp.generate_dataset(num_samples=None) 
    
    # 4. 提取特征矩阵
    print("\n正在提取多维特征...")
    
    features = [] # [N_samples, N_features]
    labels = []
    
    for item in tqdm(data):
        sents = item['sentences']
        response = item['response']
        samples = item['samples']
        item_labels = item['labels']
        
        # 1. Prompt Score (Standard)
        prompt_scores = prompt_checker.predict(sents, samples, verbose=False)
        
        # 2. Ngram Score
        ng_res = ngram_checker.predict(sents, response, samples)
        ngram_scores = ng_res['sent_level']['max_neg_logprob']
        
        # 3. Whitebox Scores (Entropy & PPL)
        entropy_scores = []
        ppl_scores = []
        for s in sents:
            ent, ppl = whitebox_analyzer.compute_uncertainty(s)
            entropy_scores.append(ent)
            ppl_scores.append(ppl)
            
        # 汇总
        for i in range(len(sents)):
            features.append([
                prompt_scores[i],      # Feature 0: SelfCheck Prompt
                ngram_scores[i],       # Feature 1: SelfCheck Ngram
                entropy_scores[i],     # Feature 2: Token Entropy (Uncertainty)
                ppl_scores[i],         # Feature 3: Perplexity
                len(sents[i])          # Feature 4: Sentence Length
            ])
            labels.append(item_labels[i])
            
    X = np.array(features)
    y = np.array(labels)
    
    # 过滤单一标签
    print(f"\n特征矩阵形状: {X.shape}, 标签分布: {np.bincount(y.astype(int))}")
    
    # 5. 训练融合模型 (Logistic Regression)
    # 使用 5折交叉验证 来评估性能，避免过拟合
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    baseline_aucs = [] # 对比 Prompt 单体
    
    print("\n开始交叉验证训练 (Random Forest)...")
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 使用随机森林 (捕捉非线性关系)
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict_proba(X_test)[:, 1]
        
        # 计算 Fusion AUC
        p, r, _ = precision_recall_curve(y_test, y_pred)
        fusion_auc = auc(r, p)
        aucs.append(fusion_auc)
        
        # 计算 Baseline AUC (Prompt Only)
        p_b, r_b, _ = precision_recall_curve(y_test, X_test[:, 0])
        base_auc = auc(r_b, p_b)
        baseline_aucs.append(base_auc)
        
        print(f"Fold {fold}: Baseline={base_auc:.4f}, Fusion={fusion_auc:.4f}")
        
        # 打印特征重要性
        if fold == 1:
            print("Feature Importances:")
            feat_names = ['Prompt', 'Ngram', 'Entropy', 'PPL', 'Length']
            for name, imp in zip(feat_names, clf.feature_importances_):
                print(f"  {name}: {imp:.4f}")
        fold += 1
        
    avg_fusion = np.mean(aucs)
    avg_base = np.mean(baseline_aucs)
    
    print("\n" + "="*50)
    print("【最终战果】")
    print(f"Single Best (Prompt): {avg_base:.4f}")
    print(f"Multi-Feature Fusion: {avg_fusion:.4f}")
    print(f"提升幅度: {avg_fusion - avg_base:+.4f}")
    print("="*50)
    
    if avg_fusion > avg_base:
        print("成功，融合模型超越了单体模型。")
        print("结论：结合内部不确定性（白盒）和外部一致性（黑盒），是检测幻觉的最佳方案。")

if __name__ == "__main__":
    run_fusion_experiment()