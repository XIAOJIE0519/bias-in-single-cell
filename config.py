"""
配置文件：定义数据集信息和分析参数
"""
import os

# 数据集配置
DATASETS = {
    'GSE159354': {
        'disease': 'scleroderma/idiopathic pulmonary fibrosis',
        'normal_count': 5,
        'folder': 'GSE159354'
    },
    'GSE132771': {
        'disease': 'scleroderma/idiopathic pulmonary fibrosis',
        'normal_count': 6,
        'folder': 'GSE132771'
    },
    'GSE173896': {
        'disease': 'chronic obstructive pulmonary disease',
        'normal_count': 6,
        'folder': 'GSE173896'
    },
    'GSE227691': {
        'disease': 'chronic obstructive pulmonary disease',
        'normal_count': 4,
        'folder': 'GSE227691'
    },
    'GSE171524': {
        'disease': 'COVID',
        'normal_count': 7,
        'folder': 'GSE171524'
    }
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER_FILE = os.path.join(BASE_DIR, 'marker-lung.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 质控参数
QC_PARAMS = {
    'min_genes': 200,           # 每个细胞最少基因数
    'min_cells': 3,             # 每个基因最少细胞数
    'max_genes': 6000,          # 每个细胞最多基因数
    'max_mito_pct': 20,         # 最大线粒体基因百分比
    'min_counts': 500,          # 每个细胞最少UMI数
    'doublet_threshold': 0.25   # doublet评分阈值
}

# 归一化和降维参数
PROCESSING_PARAMS = {
    'n_top_genes': 2000,        # 高变基因数量
    'n_pcs': 50,                # PCA主成分数量
    'n_neighbors': 15,          # UMAP邻居数
    'resolution': 0.8,          # Leiden聚类分辨率
    'min_cluster_size': 50      # 最小亚群细胞数（用于过滤）
}

# 差异表达分析参数
DE_PARAMS = {
    'method': 'wilcoxon',
    'min_pct': 0.1,
    'logfc_threshold': 0.25,
    'pval_cutoff': 0.05
}

# hdWGCNA参数
WGCNA_PARAMS = {
    'soft_power': 6,
    'min_module_size': 30,
    'merge_threshold': 0.25,
    'n_top_genes': 3000
}

# 随机种子
RANDOM_SEED = 42
