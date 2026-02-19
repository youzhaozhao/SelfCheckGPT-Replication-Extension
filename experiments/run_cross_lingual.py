import os
import sys
import torch
import numpy as np
import re
import json
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr

# 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data_loader import WikiBioDataLoader
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram

# ==============================================================================
# 0. 配置区域 - 请根据实际情况设置模型路径
# ==============================================================================
# 设置 Baichuan2 模型的本地路径。可以通过环境变量 BAICHUAN2_PATH 指定，
# 否则需要手动修改为实际路径。
LOCAL_MODEL_PATH = os.environ.get("BAICHUAN2_PATH", "/path/to/baichuan2/model")

# ==============================================================================
# 1. 模型包装器 (适配 Baichuan2)
# ==============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

class BaichuanWrapper:
    def __init__(self, model_path):
        print(f"正在加载 Baichuan2 模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=False, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.device = self.model.device
        print("模型加载完成。")

    def chat_generate(self, prompt, do_sample=False, temperature=1.0):
        """使用 Baichuan 的 Chat 模板生成"""
        messages = [{"role": "user", "content": prompt}]
        
        # Baichuan2 官方推荐的生成方式
        if do_sample:
            # 随机采样 (用于生成 Samples)
            self.model.generation_config.temperature = temperature
            self.model.generation_config.top_p = 0.8
            self.model.generation_config.do_sample = True
        else:
            # 贪婪解码 (用于 Response 和检测)
            self.model.generation_config.do_sample = False
        
        response = self.model.chat(self.tokenizer, messages)
        return response

# ==============================================================================
# 2. 中文检测器实现
# ==============================================================================

class ChineseSelfCheckNgram(SelfCheckNgram):
    def __init__(self, n: int):
        super().__init__(n=n, lowercase=False)

    def predict(self, sentences, passage, sampled_passages):
        # 中文分词处理（按字符切分）
        def char_tokenize(text):
            return " ".join(list("".join(text.split())))

        tokenized_sentences = [char_tokenize(s) for s in sentences]
        tokenized_passage = char_tokenize(passage)
        tokenized_samples = [char_tokenize(s) for s in sampled_passages]
        return super().predict(tokenized_sentences, tokenized_passage, tokenized_samples)

class BaichuanSelfCheckPrompt:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def predict(self, sentences, sampled_passages, verbose=False):
        sent_scores = []
        # 定义指令，要求模型只返回“是”或“否”
        system_instruction = "请判断下面的【待检测句子】是否被【上下文】所支持。如果支持，请回答“是”；如果不支持或矛盾，请回答“否”。只回答一个字。"
        
        for sentence in tqdm(sentences, disable=not verbose, desc="Baichuan-Prompt"):
            sample_scores = []
            for sample_passage in sampled_passages:
                try:
                    prompt = f"{system_instruction}\n\n【上下文】：{sample_passage}\n\n【待检测句子】：{sentence}\n\n答案："
                    
                    # 使用贪婪解码
                    response = self.wrapper.chat_generate(prompt, do_sample=False)
                    
                    # 解析
                    if '否' in response or '不' in response:
                        score = 1.0
                    elif '是' in response or '支持' in response:
                        score = 0.0
                    else:
                        score = 0.5
                    sample_scores.append(score)
                except:
                    sample_scores.append(0.5)
            
            sent_scores.append(np.mean(sample_scores) if sample_scores else 0.5)
        return np.array(sent_scores)

# ==============================================================================
# 3. 实验控制器
# ==============================================================================

class CrossLingualExperiment:
    def __init__(self, data_path=None):
        self.loader = WikiBioDataLoader(data_path)
        # 加载 Baichuan
        self.baichuan = BaichuanWrapper(LOCAL_MODEL_PATH)
        self.ngram_checker = ChineseSelfCheckNgram(n=1)
        self.prompt_checker = BaichuanSelfCheckPrompt(self.baichuan)

    def generate_dataset(self, num_samples=None):
        """
        num_samples: 如果为 None，则运行全量数据
        """
        # 1. 确定运行数量
        total_available = len(self.loader)
        if num_samples is None:
            target_count = total_available
            print(f"正在构建中文数据集 (全量模式: {target_count} 条)...")
        else:
            target_count = num_samples
            print(f"正在构建中文数据集 (测试模式: {target_count} 条)...")

        dataset = []
        
        def split_chinese(text):
            text = re.sub(r'([。！？])', r'\1\n', text)
            return [s.strip() for s in text.split('\n') if len(s) > 5]

        # 遍历 WikiBio 数据
        for idx in range(total_available):
            if len(dataset) >= target_count: break
            
            # 获取英文 Ref
            original_sample = self.loader.get_sample(idx)
            ref_text = original_sample.get('wiki_bio_text', '')
            if len(ref_text) < 50: continue

            try:
                # 显示进度
                if len(dataset) % 10 == 0:
                    print(f"进度: [{len(dataset)}/{target_count}]")

                # 1. 翻译 Ref -> Ground Truth
                trans_prompt = f"请将以下英文段落翻译成中文，只输出翻译结果：\n{ref_text[:500]}"
                ground_truth = self.baichuan.chat_generate(trans_prompt, do_sample=False)
                
                # 提取名字
                concept = ground_truth[:10].split('，')[0]

                # 2. 生成 Response (贪婪)
                gen_prompt = f"请用中文介绍一下{concept}。"
                response = self.baichuan.chat_generate(gen_prompt, do_sample=False)
                
                # 3. 生成 Samples (随机)
                samples = []
                for _ in range(5):
                    samp = self.baichuan.chat_generate(gen_prompt, do_sample=True, temperature=1.0)
                    samples.append(samp)

                # 4. 自动标注 (Baichuan Judge)
                sentences = split_chinese(response)
                labels = []
                for sent in sentences:
                    judge_prompt = f"事实：{ground_truth}\n\n待测：{sent}\n\n待测句子是否符合事实？回答正确或错误。"
                    ans = self.baichuan.chat_generate(judge_prompt, do_sample=False)
                    labels.append(1.0 if "错误" in ans else 0.0)

                dataset.append({
                    "sentences": sentences,
                    "response": response,
                    "samples": samples,
                    "labels": labels
                })
                
            except Exception as e:
                print(f"Skipping idx {idx}: {e}")
                continue
        
        return dataset

    def run(self):
        # 1. 生成数据 (设置为 None 以跑全量，或者设置为 238)
        data = self.generate_dataset(num_samples=None) 
        
        # 句子级数据容器
        sent_results = {
            'ngram_scores': [],
            'prompt_scores': [],
            'labels': []
        }
        
        # 段落级数据容器 (用于计算 Pearson/Spearman)
        passage_results = {
            'ngram_scores': [],
            'prompt_scores': [],
            'labels': []
        }

        # 2. 评估
        print("\n开始检测与指标计算...")
        for item in tqdm(data, desc="Processing"):
            sentences = item['sentences']
            labels = item['labels']
            
            # 过滤无效数据
            if not sentences: continue

            # --- Ngram 方法 ---
            ng_res = self.ngram_checker.predict(sentences, item['response'], item['samples'])
            scores_ng = ng_res['sent_level']['max_neg_logprob']
            
            # --- Prompt 方法 ---
            scores_pr = self.prompt_checker.predict(sentences, item['samples'], verbose=False)
            
            # 收集句子级数据
            sent_results['ngram_scores'].extend(scores_ng)
            sent_results['prompt_scores'].extend(scores_pr)
            sent_results['labels'].extend(labels)
            
            # 收集段落级数据 (取平均值)
            # 注意：如果一个段落里全是事实(0)，平均分就是0；全是幻觉(1)，平均分就是1
            if len(scores_ng) > 0:
                passage_results['ngram_scores'].append(np.mean(scores_ng))
                passage_results['prompt_scores'].append(np.mean(scores_pr))
                passage_results['labels'].append(np.mean(labels))

        # 3. 计算并打印结果
        print("\n" + "="*50)
        print(f"最终统计: 有效样本数 {len(data)} (段落), 句子数 {len(sent_results['labels'])}")
        print("="*50)

        self._print_full_metrics("Baichuan-Ngram", sent_results['ngram_scores'], sent_results['labels'], 
                               passage_results['ngram_scores'], passage_results['labels'])
        
        print("-" * 50)
        
        self._print_full_metrics("Baichuan-Prompt", sent_results['prompt_scores'], sent_results['labels'],
                               passage_results['prompt_scores'], passage_results['labels'])

    def _print_full_metrics(self, name, sent_scores, sent_labels, pass_scores, pass_labels):
        print(f"Method: {name}")
        
        # 1. 句子级别 AUC
        sent_scores = np.array(sent_scores)
        sent_labels = np.array(sent_labels)
        
        # 检查标签多样性
        if len(np.unique(sent_labels)) < 2:
            print("  [Warning] 标签单一，无法计算 AUC")
        else:
            # NonFact AUC
            precision, recall, _ = precision_recall_curve(sent_labels, sent_scores)
            auc_val = auc(recall, precision)
            
            # Factual AUC (反转分数)
            precision_f, recall_f, _ = precision_recall_curve(1-sent_labels, -sent_scores)
            auc_fact = auc(recall_f, precision_f)
            
            print(f"  [Sentence-Level]")
            print(f"    NonFact AUC-PR: {auc_val:.4f}")
            print(f"    Factual AUC-PR: {auc_fact:.4f}")

        # 2. 段落级别 Correlation
        pass_scores = np.array(pass_scores)
        pass_labels = np.array(pass_labels)
        
        if len(pass_scores) < 2:
            print("  [Warning] 段落数过少，无法计算相关性")
        else:
            pearson_corr, _ = pearsonr(pass_scores, pass_labels)
            spearman_corr, _ = spearmanr(pass_scores, pass_labels)
            
            print(f"  [Passage-Level]")
            print(f"    Pearson Correlation:  {pearson_corr:.4f}")
            print(f"    Spearman Correlation: {spearman_corr:.4f}")

if __name__ == "__main__":
    exp = CrossLingualExperiment()
    exp.run()