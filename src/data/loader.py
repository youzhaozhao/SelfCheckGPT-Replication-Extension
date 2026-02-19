import json
import os
from typing import List, Dict, Any

class WikiBioDatasetLoader:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = "./data"
        
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, "wikibio_gpt3_hallucination.json")
    
    def load_dataset(self) -> List[Dict]:
        """加载数据集"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理DatasetDict格式 - 提取evaluation split
        if isinstance(data, dict) and 'evaluation' in data:
            raw_instances = data['evaluation']
        else:
            raise ValueError(f"未知的数据集格式: {type(data)}")
        
        # 处理每个实例
        processed_instances = []
        for inst in raw_instances:
            processed_inst = self._process_instance(inst)
            processed_instances.append(processed_inst)
        
        return processed_instances
    
    def _process_instance(self, instance: Dict) -> Dict:
        """处理单个实例"""
        # 将字符串标注转换为数字
        annotation_mapping = {
            'accurate': 0.0,
            'minor_inaccurate': 0.5, 
            'major_inaccurate': 1.0
        }
        
        numeric_annotations = []
        for ann in instance['annotation']:
            if isinstance(ann, (int, float)):
                numeric_annotations.append(ann)
            elif isinstance(ann, str) and ann in annotation_mapping:
                numeric_annotations.append(annotation_mapping[ann])
            else:
                # 未知格式，设置为默认值
                print(f"警告: 未知标注格式 '{ann}'，设置为0.5")
                numeric_annotations.append(0.5)
        
        processed_instance = {
            'id': instance.get('wiki_bio_test_idx', 'unknown'),
            'gpt3_text': instance['gpt3_text'],
            'gpt3_sentences': instance['gpt3_sentences'],
            'annotation': numeric_annotations,
            'gpt3_text_samples': instance['gpt3_text_samples'],
            'wiki_bio_text': instance.get('wiki_bio_text', '')
        }
        
        return processed_instance
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        dataset = self.load_dataset()
        
        total_instances = len(dataset)
        total_sentences = sum(len(inst['gpt3_sentences']) for inst in dataset)
        total_samples = sum(len(inst['gpt3_text_samples']) for inst in dataset)
        
        # 统计标注分布
        all_annotations = []
        for inst in dataset:
            all_annotations.extend(inst['annotation'])
        
        # 根据论文的分类标准
        accurate_count = sum(1 for a in all_annotations if a == 0)
        minor_inaccurate_count = sum(1 for a in all_annotations if 0 < a < 1)
        major_inaccurate_count = sum(1 for a in all_annotations if a == 1)
        
        annotation_stats = {
            'total': len(all_annotations),
            'accurate_count': accurate_count,
            'minor_inaccurate_count': minor_inaccurate_count, 
            'major_inaccurate_count': major_inaccurate_count,
            'avg_annotation': sum(all_annotations) / len(all_annotations),
            'annotation_range': (min(all_annotations), max(all_annotations))
        }
        
        return {
            'total_instances': total_instances,
            'total_sentences': total_sentences,
            'total_samples': total_samples,
            'annotation_stats': annotation_stats
        }
    
    def validate_dataset(self) -> bool:
        """全面验证数据集"""
        try:
            dataset = self.load_dataset()
            
            print("数据集验证结果:")
            print("=" * 50)
            
            # 1. 检查实例数量
            print(f"[OK] 实例数量: {len(dataset)} (应为238)")
            
            # 2. 检查第一个实例的结构
            first_instance = dataset[0]
            required_fields = ['gpt3_text', 'gpt3_sentences', 'annotation', 'gpt3_text_samples']
            missing_fields = [field for field in required_fields if field not in first_instance]
            
            if missing_fields:
                print(f"[FAIL] 缺少字段: {missing_fields}")
                return False
            else:
                print("[OK] 所有必需字段都存在")
            
            # 3. 检查句子和标注数量是否匹配
            sentences_count = len(first_instance['gpt3_sentences'])
            annotations_count = len(first_instance['annotation'])
            if sentences_count == annotations_count:
                print(f"[OK] 句子和标注数量匹配: {sentences_count}")
            else:
                print(f"[FAIL] 句子和标注数量不匹配: {sentences_count} vs {annotations_count}")
                return False
            
            # 4. 检查采样段落数量
            samples_count = len(first_instance['gpt3_text_samples'])
            print(f"[OK] 采样段落数量: {samples_count} (应为20)")
            
            # 5. 检查标注格式
            annotations = first_instance['annotation']
            if all(isinstance(ann, (int, float)) for ann in annotations):
                print(f"[OK] 标注格式正确: 数值型 ({min(annotations)} ~ {max(annotations)})")
            else:
                print(f"[FAIL] 标注格式错误: 包含非数值类型")
                return False
            
            # 6. 检查标注值范围
            all_annotations = []
            for inst in dataset:
                all_annotations.extend(inst['annotation'])
            
            unique_values = set(all_annotations)
            expected_values = {0.0, 0.5, 1.0}
            if unique_values.issubset(expected_values):
                print(f"[OK] 标注值范围正确: {sorted(unique_values)}")
            else:
                print(f"[WARN] 标注值范围异常: {sorted(unique_values)} (期望: {sorted(expected_values)})")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] 验证过程中出错: {e}")
            return False
    
    def print_dataset_summary(self):
        """打印数据集摘要"""
        info = self.get_dataset_info()
        
        print("=" * 60)
        print("WikiBio GPT-3 Hallucination Dataset 完整摘要")
        print("=" * 60)
        print(f"实例总数: {info['total_instances']}")
        print(f"句子总数: {info['total_sentences']}")
        print(f"采样段落总数: {info['total_samples']}")
        
        stats = info['annotation_stats']
        print(f"\n标注统计 (基于{stats['total']}个句子标注):")
        print(f"  - 准确句子 (0.0): {stats['accurate_count']:4d} ({stats['accurate_count']/stats['total']*100:5.1f}%)")
        print(f"  - 次要不准确 (0.5): {stats['minor_inaccurate_count']:4d} ({stats['minor_inaccurate_count']/stats['total']*100:5.1f}%)") 
        print(f"  - 主要不准确 (1.0): {stats['major_inaccurate_count']:4d} ({stats['major_inaccurate_count']/stats['total']*100:5.1f}%)")
        print(f"  - 平均标注值: {stats['avg_annotation']:.3f}")
        print(f"  - 标注范围: {stats['annotation_range']}")
        
        # 与论文数据对比
        print(f"\n与论文数据对比:")
        paper_accurate = 516
        paper_minor = 631  
        paper_major = 761
        paper_total = 1908
        
        print(f"  - 论文准确: {paper_accurate} ({paper_accurate/paper_total*100:.1f}%)")
        print(f"  - 论文次要: {paper_minor} ({paper_minor/paper_total*100:.1f}%)")
        print(f"  - 论文主要: {paper_major} ({paper_major/paper_total*100:.1f}%)")
        print("=" * 60)

# 测试函数
def test_dataset():
    """测试数据集加载和验证"""
    loader = WikiBioDatasetLoader()
    
    print("开始验证数据集...")
    
    # 验证数据集结构
    is_valid = loader.validate_dataset()
    
    if is_valid:
        print("\n[PASS] 数据集验证通过！")
        
        # 显示详细摘要
        loader.print_dataset_summary()
        
        # 显示一个完整实例作为示例
        dataset = loader.load_dataset()
        sample = dataset[0]
        
        print(f"\n示例实例 (ID: {sample['id']}):")
        print(f"生成文本预览: {sample['gpt3_text'][:100]}...")
        print(f"句子数量: {len(sample['gpt3_sentences'])}")
        print(f"前3个句子:")
        for i, (sent, ann) in enumerate(zip(sample['gpt3_sentences'][:3], sample['annotation'][:3])):
            print(f"  {i+1}. [{ann}] {sent[:50]}...")
        print(f"采样段落数量: {len(sample['gpt3_text_samples'])}")
        
        return True
    else:
        print("\n[FAIL] 数据集验证失败，请检查数据文件。")
        return False

if __name__ == "__main__":
    test_dataset()