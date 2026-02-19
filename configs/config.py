import os
import sys

def setup_paths():
    """
    设置项目路径，返回项目根目录
    """
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 计算项目根目录（configs 的父目录）
    project_root = os.path.dirname(current_dir)
    
    # 添加到 Python 路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 定义其他重要路径
    paths = {
        'project_root': project_root,
        'data_dir': os.path.join(project_root, 'data'),
        'scripts_dir': os.path.join(project_root, 'scripts'),
        'experiments_dir': os.path.join(project_root, 'experiments'),
        'models_dir': os.path.join(project_root, 'models'),
        'results_dir': os.path.join(project_root, 'results'),
        'tests_dir': os.path.join(project_root, 'tests'),
    }
    
    # 创建必要的目录
    for key, path in paths.items():
        if key.endswith('_dir'):
            os.makedirs(path, exist_ok=True)
    
    return paths

# 执行路径设置
PATHS = setup_paths()

# 导出常用路径
PROJECT_ROOT = PATHS['project_root']
DATA_DIR = PATHS['data_dir']
SCRIPTS_DIR = PATHS['scripts_dir']
EXPERIMENTS_DIR = PATHS['experiments_dir']
MODELS_DIR = PATHS['models_dir']
RESULTS_DIR = PATHS['results_dir']
TESTS_DIR = PATHS['tests_dir']

print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据目录: {DATA_DIR}")