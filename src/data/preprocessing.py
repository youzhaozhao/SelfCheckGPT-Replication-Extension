import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import WikiBioDatasetLoader

def create_final_dataset():
    """创建最终处理好的数据集文件"""
    
    loader = WikiBioDatasetLoader()
    
    print("创建最终数据集文件...")
    
    try:
        # 加载并处理数据
        dataset = loader.load_dataset()
        
        # 验证数据质量
        if not loader.validate_dataset():
            print("数据验证失败，停止处理。")
            return
        
        # 保存处理后的数据集
        output_path = os.path.join(loader.data_dir, "wikibio_processed_final.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f" 最终数据集已保存: {output_path}")
        
        # 显示统计信息
        loader.print_dataset_summary()
        
        # 创建数据集的简化版本用于快速测试
        test_dataset = dataset[:10]  # 前10个实例用于测试
        test_path = os.path.join(loader.data_dir, "wikibio_test_subset.json")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, indent=2, ensure_ascii=False)
        
        print(f" 测试子集已保存: {test_path} (10个实例)")
        
        return output_path
        
    except Exception as e:
        print(f" 创建最终数据集时出错: {e}")
        return None

if __name__ == "__main__":
    create_final_dataset()