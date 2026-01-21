"""
数据加载和预处理模块
"""
import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_h5_files(dataset_folder, dataset_id):
    """
    加载指定文件夹下的所有.h5文件
    
    Parameters:
    -----------
    dataset_folder : str
        数据集文件夹路径
    dataset_id : str
        数据集ID（如GSE159354）
    
    Returns:
    --------
    adata : AnnData
        合并后的AnnData对象
    """
    h5_files = list(Path(dataset_folder).glob('*.h5'))
    
    if len(h5_files) == 0:
        raise ValueError(f"在 {dataset_folder} 中未找到.h5文件")
    
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset_id}")
    print(f"找到 {len(h5_files)} 个.h5文件")
    print(f"{'='*60}")
    
    adatas = []
    
    for h5_file in sorted(h5_files):
        sample_id = h5_file.stem
        print(f"\n读取: {sample_id}")
        
        try:
            # 尝试使用10x_h5格式读取
            adata = sc.read_10x_h5(h5_file)
            
            # 添加样本信息
            adata.obs['sample_id'] = sample_id
            adata.obs['dataset_id'] = dataset_id
            
            print(f"  细胞数: {adata.n_obs}")
            print(f"  基因数: {adata.n_vars}")
            
            adatas.append(adata)
            
        except Exception as e:
            print(f"  读取失败: {str(e)}")
            continue
    
    if len(adatas) == 0:
        raise ValueError(f"数据集 {dataset_id} 中没有成功读取任何文件")
    
    # 合并所有样本
    print(f"\n合并 {len(adatas)} 个样本...")
    adata_combined = adatas[0].concatenate(
        adatas[1:],
        batch_key='sample_id',
        batch_categories=[a.obs['sample_id'][0] for a in adatas],
        index_unique='-'
    )
    
    # 确保dataset_id正确传递
    adata_combined.obs['dataset_id'] = dataset_id
    
    print(f"合并后总细胞数: {adata_combined.n_obs}")
    print(f"合并后总基因数: {adata_combined.n_vars}")
    
    return adata_combined


def load_all_datasets(datasets_config, base_dir):
    """
    加载所有数据集
    
    Parameters:
    -----------
    datasets_config : dict
        数据集配置字典
    base_dir : str
        基础目录路径
    
    Returns:
    --------
    adata_all : AnnData
        合并所有数据集的AnnData对象
    """
    all_adatas = []
    
    for dataset_id, config in datasets_config.items():
        folder_path = os.path.join(base_dir, config['folder'])
        
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在，跳过")
            continue
        
        try:
            adata = load_h5_files(folder_path, dataset_id)
            
            # 添加疾病信息
            adata.obs['disease'] = config['disease']
            adata.obs['condition'] = 'normal'  # 假设所有样本都是normal
            
            all_adatas.append(adata)
            
        except Exception as e:
            print(f"加载数据集 {dataset_id} 时出错: {str(e)}")
            continue
    
    if len(all_adatas) == 0:
        raise ValueError("没有成功加载任何数据集")
    
    # 合并所有数据集
    print(f"\n{'='*60}")
    print(f"合并所有数据集...")
    print(f"{'='*60}")
    
    adata_all = all_adatas[0].concatenate(
        all_adatas[1:],
        batch_key='dataset_id',
        batch_categories=[a.obs['dataset_id'][0] for a in all_adatas],
        index_unique='-'
    )
    
    print(f"\n最终数据集统计:")
    print(f"  总细胞数: {adata_all.n_obs}")
    print(f"  总基因数: {adata_all.n_vars}")
    print(f"  数据集数: {adata_all.obs['dataset_id'].nunique()}")
    print(f"  样本数: {adata_all.obs['sample_id'].nunique()}")
    
    return adata_all


def load_marker_genes(marker_file):
    """
    加载marker基因注释文件
    
    Parameters:
    -----------
    marker_file : str
        marker基因Excel文件路径
    
    Returns:
    --------
    marker_dict : dict
        细胞类型到marker基因列表的字典
    """
    print(f"\n{'='*60}")
    print(f"加载marker基因注释")
    print(f"{'='*60}")
    
    # 读取Excel文件
    df = pd.read_excel(marker_file)
    
    marker_dict = {}
    
    # 第一列是细胞类型名称
    for idx, row in df.iterrows():
        cell_type = row.iloc[0]
        
        # 获取该行的所有marker基因（去除NaN）
        markers = [str(gene).strip() for gene in row.iloc[1:] 
                  if pd.notna(gene) and str(gene).strip() != '']
        
        if markers:
            marker_dict[cell_type] = markers
            print(f"  {cell_type}: {len(markers)} 个marker基因")
    
    print(f"\n总共加载 {len(marker_dict)} 种细胞类型")
    
    return marker_dict
