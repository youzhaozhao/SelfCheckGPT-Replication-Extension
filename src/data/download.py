import os
import json
from datasets import load_dataset

def download_wikibio_dataset(cache_dir="./data"):
    """
    从 HuggingFace 下载 WikiBio GPT-3 Hallucination 数据集。

    参数:
        cache_dir (str): 数据集缓存及保存的目录，默认为当前目录下的 'data' 文件夹。

    返回:
        dict: 包含训练、验证和测试集的数据字典，若失败则返回 None。
    """
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)

    print("正在从 HuggingFace 下载 WikiBio GPT-3 Hallucination 数据集...")

    try:
        # 加载数据集
        dataset = load_dataset(
            "potsawee/wiki_bio_gpt3_hallucination",
            cache_dir=cache_dir
        )

        print(f"数据集加载成功！")
        print(f"数据集结构: {dataset}")

        # 保存到本地 JSON 文件
        output_path = os.path.join(cache_dir, "wikibio_gpt3_hallucination.json")
        dataset_dict = {}
        for split in dataset.keys():
            dataset_dict[split] = dataset[split].to_list()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        print(f"数据集已保存到: {output_path}")
        return dataset_dict

    except Exception as e:
        print(f"从 HuggingFace 下载时出错: {e}")
        print("尝试备用下载方式...")
        return download_dataset_alternative(cache_dir)

def download_dataset_alternative(cache_dir="./data"):
    """
    备用下载方式：从 Google Drive 下载数据集。

    参数:
        cache_dir (str): 数据集保存的目录。

    返回:
        dict: 数据集字典，若失败则返回 None。
    """
    try:
        import gdown
    except ImportError:
        print("未安装 gdown，请运行 `pip install gdown` 以使用备用下载。")
        return None

    # Google Drive 文件 ID（来自数据集官方仓库说明）
    file_id = "1AyQ7u9nYlZgUZLm5JBDx6cFFWB__EsNv"
    url = f"https://drive.google.com/uc?id={file_id}"

    output_path = os.path.join(cache_dir, "wikibio_gpt3_hallucination.json")
    print("正在从 Google Drive 下载数据集...")

    try:
        gdown.download(url, output_path, quiet=False)
        print(f"数据集下载完成: {output_path}")

        # 验证下载
        with open(output_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"数据集验证成功，包含 {len(dataset)} 个实例")
        return dataset

    except Exception as e:
        print(f"备用下载失败: {e}")
        return None

def validate_dataset_structure(dataset):
    """
    验证数据集结构是否符合预期。

    参数:
        dataset (dict): 数据集字典（应包含 'train' 等 split 或直接为列表）。

    返回:
        bool: 验证是否通过。
    """
    print("\n验证数据集结构...")

    if isinstance(dataset, dict) and 'train' in dataset:
        instances = dataset['train']
    elif isinstance(dataset, list):
        instances = dataset
    else:
        print("未知的数据集格式")
        return False

    # 检查第一个实例的关键字段
    sample_instance = instances[0]
    required_fields = ['gpt3_text', 'gpt3_sentences', 'annotation', 'gpt3_text_samples']

    missing_fields = [field for field in required_fields if field not in sample_instance]
    if missing_fields:
        print(f"缺少必要字段: {missing_fields}")
        return False

    print("✓ gpt3_text:", len(sample_instance['gpt3_text']), "字符")
    print("✓ gpt3_sentences:", len(sample_instance['gpt3_sentences']), "个句子")
    print("✓ annotation:", len(sample_instance['annotation']), "个标注")
    print("✓ gpt3_text_samples:", len(sample_instance['gpt3_text_samples']), "个采样段落")

    # 验证标注范围
    annotations = sample_instance['annotation']
    print("✓ 标注范围:", f"{min(annotations)} ~ {max(annotations)}")

    return True

if __name__ == "__main__":
    # 可自定义缓存路径（例如通过环境变量或直接修改此处）
    cache_dir = "./data"  # 可根据需要改为绝对路径或从环境变量读取
    dataset = download_wikibio_dataset(cache_dir)

    if dataset is not None:
        is_valid = validate_dataset_structure(dataset)
        if is_valid:
            print("\n 数据集准备完成！可以开始实验了。")
        else:
            print("\n 数据集结构验证失败，请检查数据格式。")
    else:
        print("\n 数据集下载失败，请检查网络连接或手动下载。")