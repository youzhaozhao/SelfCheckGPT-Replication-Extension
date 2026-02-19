import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=UserWarning)   
warnings.filterwarnings('ignore', category=DeprecationWarning)  
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore",
    message=r".*Forcing disable 'CUTLASS' backend as it is not supported in.*",
    category=UserWarning,
    module="torch._inductor.utils"
)
torch._dynamo.config.suppress_errors = True

import json
from scipy.stats import pearsonr, spearmanr  
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from torchmetrics import PearsonCorrCoef
from typing import List
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from scripts.data_loader import WikiBioDataLoader
from selfcheckgpt.modeling_selfcheck import (
    SelfCheckNLI, SelfCheckBERTScore, SelfCheckNgram, 
    SelfCheckMQAG, SelfCheckPrompt
)

class ReproductionExperiment:
    def __init__(self, data_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.loader = WikiBioDataLoader(data_path)
        self.device = device
        self.results = {}
        
    def run_selfcheck_nli(self):
        """复现SelfCheckNLI方法"""
        print("=== 运行 SelfCheckNLI ===")
        checker = SelfCheckNLI(device=self.device)
        
        all_scores, all_labels = [], []
        
        for idx in tqdm(range(min(50, len(self.loader)))):  
            sentences = self.loader.get_sentences(idx)
            sampled_passages = self.loader.get_sampled_passages(idx)
            labels = self.loader.get_labels(idx)
            
            if len(sentences) != len(labels):
                continue
                
            try:
                scores = checker.predict(sentences, sampled_passages)
                all_scores.extend(scores)
                all_labels.extend(labels)
            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue
        
        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckNLI', all_scores, all_labels)

        self.results['SelfCheckNLI'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels,
            'simple_auc_pr': self._compute_metrics(all_scores, all_labels)['auc_pr']  
        }
        return paper_metrics
    
    def run_selfcheck_bertscore(self):
        """复现SelfCheckBERTScore方法"""
        print("=== 运行 SelfCheckBERTScore ===")
        checker = SelfCheckBERTScore()

        all_scores, all_labels = [], []
        valid_samples = 0

        for idx in tqdm(range(min(50, len(self.loader)))):
            sentences = self.loader.get_sentences(idx)
            sampled_passages = self.loader.get_sampled_passages(idx)
            labels = self.loader.get_labels(idx)

            if len(sentences) != len(labels):
                continue

            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"跳过样本 {idx}：标签单一 {unique_labels}")
                continue

            try:
                scores = checker.predict(sentences, sampled_passages)
                all_scores.extend(scores)
                all_labels.extend(labels)
                valid_samples += 1
            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue

        if len(all_scores) == 0:
            print("没有有效的样本可用于BERTScore评估")
            return None

        print(f"使用 {valid_samples} 个有效样本进行BERTScore评估")
        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckBERTScore', all_scores, all_labels)

        self.results['SelfCheckBERTScore'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels
        }
        return paper_metrics
    
    def run_selfcheck_ngram(self, n: int = 1):
        """复现SelfCheckNgram方法"""
        print(f"=== 运行 SelfCheck{ n }gram ===")
        checker = SelfCheckNgram(n=n)
        
        all_scores, all_labels = [], []
        
        for idx in tqdm(range(min(30, len(self.loader)))):  
            sentences = self.loader.get_sentences(idx)
            passage = self.loader.get_sample(idx).get('gpt3_text', '') 
            sampled_passages = self.loader.get_sampled_passages(idx)
            labels = self.loader.get_labels(idx)
            
            if len(sentences) != len(labels):
                continue
                
            try:
                result = checker.predict(sentences, passage, sampled_passages)
                scores = result['sent_level']['max_neg_logprob']  
                all_scores.extend(scores)
                all_labels.extend(labels)
            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue
        
        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckNgram', all_scores, all_labels)

        self.results['SelfCheckNgram'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels
        }
        return paper_metrics
    
    def _compute_metrics(self, scores: List[float], labels: List[int]):
        """计算评估指标，处理各种边界情况"""
        if len(scores) == 0 or len(labels) == 0:
            print("错误：分数或标签为空数组")
            return {
                'auc_pr': 0.5,
                'pearson': 0.0,
                'num_samples': 0,
                'warning': "空数据"
            }

        scores = np.array(scores)
        labels = np.array(labels)

        unique_labels = np.unique(labels)
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

        if len(unique_labels) < 2:
            warning_msg = f"单一类别: {unique_labels[0]}" if len(unique_labels) == 1 else "无标签数据"
            print(f"警告：{warning_msg}，无法计算AUC-PR")
            return {
                'auc_pr': 0.5,
                'pearson': 0.0,
                'num_samples': len(scores),
                'warning': warning_msg
            }

        # AUC-PR
        try:
            precision, recall, _ = precision_recall_curve(labels, scores)
            auc_pr = auc(recall, precision)
        except Exception as e:
            print(f"计算AUC-PR失败: {e}")
            auc_pr = 0.5

        # Pearson相关系数
        pearson_metric = PearsonCorrCoef()
        try:
            pearson = pearson_metric(
                torch.tensor(scores, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32)
            ).item()
        except:
            pearson = 0.0

        return {
            'auc_pr': auc_pr,
            'pearson': pearson,
            'num_samples': len(scores)
        }
    
    def run_all_methods(self):
        """运行所有方法，使用论文评估标准"""
        print("开始复现所有SelfCheckGPT方法（使用论文评估标准）...")

        # 获取所有样本索引
        all_samples = list(range(len(self.loader)))
        print(f"总样本数: {len(all_samples)}")

        # 运行所有方法
        self.run_selfcheck_ngram_with_samples(all_samples)
        self.run_selfcheck_nli_with_samples(all_samples)
        self.run_selfcheck_prompt_with_samples(all_samples)
        self.run_selfcheck_bertscore_with_samples(all_samples)
        self.run_selfcheck_mqag_with_samples(all_samples) 

        return self.results
    
    def evaluate_with_paper_metrics(self, method_name: str, scores: List[float], labels: List[float], passage_indices: List[int] = None):
        """
        按照原论文的评估标准计算三个任务：
        - NonFact: major_inaccurate(1) + minor_inaccurate(0.5) vs accurate(0)
        - NonFact*: 只包含major_inaccurate(1) vs 其他(0+0.5)
        - Factual: accurate(0) vs non-factual(0.5+1.0)
        同时计算段落级别的皮尔逊和斯皮尔曼相关系数
        """
        scores = np.array(scores)
        labels = np.array(labels)

        # Task 1: NonFact (检测所有非事实性句子)
        nonfact_binary = (labels > 0).astype(int)
        nonfact_auc_pr = self._compute_auc_pr(scores, nonfact_binary)

        # Task 2: NonFact* (只检测major_inaccurate)
        nonfact_star_binary = (labels == 1.0).astype(int)
        nonfact_star_auc_pr = self._compute_auc_pr(scores, nonfact_star_binary)

        # Task 3: Factual (检测事实性句子) - 关键修复！
        factual_binary = (labels == 0.0).astype(int)
        factual_auc_pr = self._compute_auc_pr(1 - scores, factual_binary)  # 使用 1 - scores

        # 计算段落级别的相关性（如果提供了段落索引）
        pearson_corr, spearman_corr = 0.0, 0.0
        if passage_indices is not None and len(passage_indices) == len(scores):
            pearson_corr, spearman_corr = self._compute_passage_correlations(scores, labels, passage_indices)

        return {
            'NonFact': nonfact_auc_pr,
            'NonFact*': nonfact_star_auc_pr,
            'Factual': factual_auc_pr,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        }

    def _compute_auc_pr(self, scores: np.ndarray, binary_labels: np.ndarray):
        """计算AUC-PR的辅助函数"""
        # 确保是二分类问题
        unique_labels = np.unique(binary_labels)
        if len(unique_labels) < 2:
            print(f"警告: 只有单一类别 {unique_labels}，无法计算AUC-PR")
            return 0.5

        try:
            precision, recall, _ = precision_recall_curve(binary_labels, scores)
            return auc(recall, precision)
        except Exception as e:
            print(f"计算AUC-PR失败: {e}")
            return 0.5
        
        
    def _compute_passage_correlations(self, scores: np.ndarray, labels: np.ndarray, passage_indices: List[int]):
        """
        计算段落级别的皮尔逊和斯皮尔曼相关系数
        """
        # 按段落聚合分数和标签
        passage_scores = {}
        passage_labels = {}

        for score, label, passage_idx in zip(scores, labels, passage_indices):
            if passage_idx not in passage_scores:
                passage_scores[passage_idx] = []
                passage_labels[passage_idx] = []
            passage_scores[passage_idx].append(score)
            passage_labels[passage_idx].append(label)

        # 计算每个段落的平均分数和平均标签
        avg_passage_scores = []
        avg_passage_labels = []

        for passage_idx in sorted(passage_scores.keys()):
            passage_score = np.mean(passage_scores[passage_idx])
            passage_label = np.mean(passage_labels[passage_idx])

            avg_passage_scores.append(passage_score)
            avg_passage_labels.append(passage_label)

        # 计算相关系数
        try:
            if len(avg_passage_scores) >= 2:  # 需要至少2个点计算相关性
                pearson_corr, _ = pearsonr(avg_passage_scores, avg_passage_labels)
                spearman_corr, _ = spearmanr(avg_passage_scores, avg_passage_labels)
            else:
                pearson_corr, spearman_corr = 0.0, 0.0
                print(f"段落数不足({len(avg_passage_scores)})，无法计算相关性")
        except Exception as e:
            print(f"计算相关性失败: {e}")
            pearson_corr, spearman_corr = 0.0, 0.0

        return pearson_corr, spearman_corr


    def run_selfcheck_bertscore_with_samples(self, sample_indices: List[int]):
        """使用指定样本运行BERTScore"""
        print("=== 运行 SelfCheckBERTScore ===")
        checker = SelfCheckBERTScore()

        all_scores, all_labels, all_passage_indices = [], [], []
        processed_count = 0

        for idx in tqdm(sample_indices):
            try:
                sentences = self.loader.get_sentences(idx)
                sampled_passages = self.loader.get_sampled_passages(idx)
                labels = self.loader.get_labels(idx)

                # 调试信息
                print(f"样本 {idx}: 句子数={len(sentences)}, 标签数={len(labels)}, 采样段落数={len(sampled_passages)}")

                # 检查数据有效性
                if len(sentences) == 0 or len(labels) == 0 or len(sampled_passages) == 0:
                    print(f"  跳过样本 {idx}：数据不完整")
                    continue

                if len(sentences) != len(labels):
                    print(f"  跳过样本 {idx}：句子数({len(sentences)})与标签数({len(labels)})不匹配")
                    continue

                # 运行BERTScore
                scores = checker.predict(sentences, sampled_passages)

                # 检查返回的分数
                if len(scores) == 0:
                    print(f"  跳过样本 {idx}：BERTScore返回空分数")
                    continue

                all_scores.extend(scores)
                all_labels.extend(labels)
                all_passage_indices.extend([idx] * len(scores)) 
                processed_count += 1

            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue

        print(f"成功处理 {processed_count} 个样本")
        print(f"总分数数量: {len(all_scores)}, 总标签数量: {len(all_labels)}")

        if len(all_scores) == 0:
            print("没有有效的BERTScore结果")
            return None

        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckBERTScore', all_scores, all_labels, all_passage_indices)

        self.results['SelfCheckBERTScore'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels,
            'passage_indices': all_passage_indices
        }
        return paper_metrics
    
    def run_selfcheck_nli_with_samples(self, sample_indices: List[int]):
        """使用指定样本运行SelfCheckNLI"""
        print("=== 运行 SelfCheckNLI ===")
        checker = SelfCheckNLI(device=self.device)

        all_scores, all_labels, all_passage_indices = [], [], [] 
        processed_count = 0

        for idx in tqdm(sample_indices):
            try:
                sentences = self.loader.get_sentences(idx)
                sampled_passages = self.loader.get_sampled_passages(idx)
                labels = self.loader.get_labels(idx)

                # 检查数据有效性
                if len(sentences) == 0 or len(labels) == 0 or len(sampled_passages) == 0:
                    continue

                if len(sentences) != len(labels):
                    continue

                # 运行NLI
                scores = checker.predict(sentences, sampled_passages)

                if len(scores) == 0:
                    continue

                all_scores.extend(scores)
                all_labels.extend(labels)
                all_passage_indices.extend([idx] * len(scores)) 
                processed_count += 1

            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue

        print(f"成功处理 {processed_count} 个样本用于NLI")

        if len(all_scores) == 0:
            print("没有有效的NLI结果")
            return None

        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckNLI', all_scores, all_labels, all_passage_indices)

        self.results['SelfCheckNLI'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels,
            'passage_indices': all_passage_indices 
        }
        return paper_metrics

    def run_selfcheck_ngram_with_samples(self, sample_indices: List[int]):
        """使用指定样本运行SelfCheckNgram"""
        print("=== 运行 SelfCheckNgram ===")
        checker = SelfCheckNgram(n=1)

        all_scores, all_labels, all_passage_indices = [], [], [] 
        processed_count = 0

        for idx in tqdm(sample_indices):
            try:
                sentences = self.loader.get_sentences(idx)
                passage = self.loader.get_sample(idx).get('gpt3_text', '')
                sampled_passages = self.loader.get_sampled_passages(idx)
                labels = self.loader.get_labels(idx)

                # 检查数据有效性
                if len(sentences) == 0 or len(labels) == 0 or len(sampled_passages) == 0:
                    continue

                if len(sentences) != len(labels):
                    continue

                # 运行Ngram
                result = checker.predict(sentences, passage, sampled_passages)
                scores = result['sent_level']['max_neg_logprob']  # 使用平均负对数概率作为分数

                if len(scores) == 0:
                    continue

                all_scores.extend(scores)
                all_labels.extend(labels)
                all_passage_indices.extend([idx] * len(scores)) 
                processed_count += 1

            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                continue

        print(f"成功处理 {processed_count} 个样本用于Ngram")

        if len(all_scores) == 0:
            print("没有有效的Ngram结果")
            return None

        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckNgram', all_scores, all_labels, all_passage_indices)

        self.results['SelfCheckNgram'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels
        }
        return paper_metrics
    
    
    def run_selfcheck_mqag_with_samples(self, sample_indices: List[int]):
        """使用SelfCheckMQAG接口"""
        print("=== 运行 SelfCheckMQAG ===")
        checker = SelfCheckMQAG(device=self.device)

        all_scores, all_labels, all_passage_indices = [], [], [] 
        processed_count = 0

        for idx in tqdm(sample_indices):
            try:
                sample = self.loader.get_sample(idx)
                sentences = self.loader.get_sentences(idx)
                passage = sample.get('gpt3_text', '')  # 主要段落
                sampled_passages = self.loader.get_sampled_passages(idx)  # 采样段落
                labels = self.loader.get_labels(idx)

                # 检查数据有效性
                if len(sentences) == 0 or len(labels) == 0 or not passage:
                    print(f"  跳过样本 {idx}：句子/标签/主要段落为空")
                    continue
                if len(sentences) != len(labels):
                    print(f"  跳过样本 {idx}：句子数({len(sentences)})与标签数({len(labels)})不匹配")
                    continue
                if len(sampled_passages) == 0:
                    print(f"  跳过样本 {idx}：采样段落为空")
                    continue

                # 使用正确的SelfCheckMQAG接口
                scores = checker.predict(
                    sentences=sentences,
                    passage=passage,
                    sampled_passages=sampled_passages,
                    num_questions_per_sent=5,  # 每个句子生成5个问题
                    scoring_method='bayes_with_alpha',  # 使用论文方法
                    beta1=0.8, beta2=0.8  # 论文参数
                )

                # 检查返回的分数
                if len(scores) != len(sentences):
                    print(f"  跳过样本 {idx}：返回分数数量({len(scores)})与句子数量({len(sentences)})不匹配")
                    continue

                all_scores.extend(scores)
                all_labels.extend(labels)
                all_passage_indices.extend([idx] * len(scores)) 
                processed_count += 1
                print(f"  样本 {idx} 处理成功：{len(sentences)} 个句子，分数范围: {min(scores):.3f} ~ {max(scores):.3f}")

            except Exception as e:
                print(f"样本 {idx} 处理失败: {str(e)[:200]}") 
                continue

        print(f"成功处理 {processed_count} 个样本用于MQAG")
        print(f"总句子数：{len(all_scores)}, 总标签数：{len(all_labels)}")

        if len(all_scores) == 0:
            print("没有有效的MQAG结果")
            return None

        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckMQAG', all_scores, all_labels, all_passage_indices)

        self.results['SelfCheckMQAG'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels,
            'passage_indices': all_passage_indices
        }
        return paper_metrics
    
    def run_selfcheck_prompt_with_samples(self, sample_indices: List[int]):
        """使用指定样本运行SelfCheckPrompt"""
        print("=== 运行 SelfCheckPrompt ===")
        checker = SelfCheckPrompt(device=self.device)

        all_scores, all_labels, all_passage_indices = [], [], [] 
        processed_count = 0

        for idx in tqdm(sample_indices):
            try:
                sentences = self.loader.get_sentences(idx)
                sampled_passages = self.loader.get_sampled_passages(idx)
                labels = self.loader.get_labels(idx)

                # 检查数据有效性
                if len(sentences) == 0 or len(labels) == 0 or len(sampled_passages) == 0:
                    continue

                if len(sentences) != len(labels):
                    continue

                # 运行Prompt方法
                scores = checker.predict(sentences, sampled_passages, verbose=False)

                if len(scores) == 0:
                    continue

                all_scores.extend(scores)
                all_labels.extend(labels)
                all_passage_indices.extend([idx] * len(scores)) 
                processed_count += 1

            except Exception as e:
                print(f"样本 {idx} Prompt处理失败: {e}")
                continue

        print(f"成功处理 {processed_count} 个样本用于Prompt")

        if len(all_scores) == 0:
            print("没有有效的Prompt结果")
            return None

        paper_metrics = self.evaluate_with_paper_metrics('SelfCheckPrompt', all_scores, all_labels, all_passage_indices)

        self.results['SelfCheckPrompt'] = {
            'paper_metrics': paper_metrics,
            'scores': all_scores,
            'labels': all_labels,
            'passage_indices': all_passage_indices  
        }
        return paper_metrics

    def _find_diverse_samples(self, num_samples: int = 50):
        """找到标签多样的样本（使用原始标签0.0, 0.5, 1.0）"""
        diverse_samples = []

        for idx in range(len(self.loader)):
            labels = self.loader.get_labels(idx)
            unique_labels = set(labels)

            # 检查是否有至少两种不同的标签类型
            if len(unique_labels) >= 2:  
                diverse_samples.append(idx)

            if len(diverse_samples) >= num_samples:
                break

        # 如果多样样本不足，补充随机样本
        if len(diverse_samples) < num_samples:
            remaining = num_samples - len(diverse_samples)
            all_indices = list(range(len(self.loader)))
            import random
            additional_samples = random.sample(all_indices, remaining)
            diverse_samples.extend(additional_samples)

        return diverse_samples

    
    def save_results(self, output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")
    
    def print_summary(self):
        """打印结果摘要（按照论文标准）"""
        print("\n" + "="*50)
        print("复现实验结果摘要（按照论文标准）")
        print("="*50)

        for method, result in self.results.items():
            if 'paper_metrics' in result:
                metrics = result['paper_metrics']
                print(f"{method}:")
                print(f"  句子级别:")
                print(f"    NonFact AUC-PR: {metrics['NonFact']:.4f}")
                print(f"    NonFact* AUC-PR: {metrics['NonFact*']:.4f}") 
                print(f"    Factual AUC-PR: {metrics['Factual']:.4f}")
                print(f"  段落级别:")
                print(f"    皮尔逊相关系数: {metrics['Pearson']:.4f}")
                print(f"    斯皮尔曼相关系数: {metrics['Spearman']:.4f}")
                print(f"  样本数: {len(result['scores'])}")
                print()

    
if __name__ == "__main__":
    # 运行复现实验
    experiment = ReproductionExperiment()
    results = experiment.run_all_methods()
    experiment.print_summary()
    experiment.save_results('reproduction_results.json')