"""
质控和预处理模块
"""
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_qc_metrics(adata):
    """
    计算质控指标
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    
    Returns:
    --------
    adata : AnnData
        添加了质控指标的AnnData对象
    """
    print(f"\n{'='*60}")
    print("计算质控指标")
    print(f"{'='*60}")
    
    # 计算线粒体基因百分比
    # 尝试多种线粒体基因命名方式
    adata.var['mt'] = (
        adata.var_names.str.startswith('MT-') | 
        adata.var_names.str.startswith('mt-') |
        adata.var_names.str.startswith('Mt-')
    )
    
    # 计算核糖体基因百分比
    adata.var['ribo'] = (
        adata.var_names.str.startswith('RPS') | 
        adata.var_names.str.startswith('RPL')
    )
    
    # 计算血红蛋白基因百分比
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')
    
    # 使用scanpy计算QC指标
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo', 'hb'],
        percent_top=None,
        log1p=False,
        inplace=True
    )
    
    print(f"线粒体基因数: {adata.var['mt'].sum()}")
    print(f"核糖体基因数: {adata.var['ribo'].sum()}")
    print(f"血红蛋白基因数: {adata.var['hb'].sum()}")
    
    # 重命名列以便于使用
    adata.obs['n_counts'] = adata.obs['total_counts']
    adata.obs['n_genes'] = adata.obs['n_genes_by_counts']
    adata.obs['pct_counts_mt'] = adata.obs['pct_counts_mt']
    
    return adata


def plot_qc_metrics(adata, output_dir):
    """
    绘制质控指标分布图
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    """
    print("\n绘制质控指标分布图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 每个数据集的统计
    for dataset in adata.obs['dataset_id'].unique():
        mask = adata.obs['dataset_id'] == dataset
        
        axes[0, 0].hist(adata.obs.loc[mask, 'n_counts'], bins=100, alpha=0.5, label=dataset)
        axes[0, 1].hist(adata.obs.loc[mask, 'n_genes'], bins=100, alpha=0.5, label=dataset)
        axes[0, 2].hist(adata.obs.loc[mask, 'pct_counts_mt'], bins=100, alpha=0.5, label=dataset)
    
    axes[0, 0].set_xlabel('Total counts')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('UMI counts distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_xlabel('Number of genes')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title('Gene counts distribution')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    axes[0, 2].set_xlabel('Mitochondrial %')
    axes[0, 2].set_ylabel('Number of cells')
    axes[0, 2].set_title('Mitochondrial percentage distribution')
    axes[0, 2].legend()
    axes[0, 2].set_yscale('log')
    
    # 散点图
    sc.pl.scatter(adata, x='n_counts', y='n_genes', color='pct_counts_mt', 
                  ax=axes[1, 0], show=False)
    sc.pl.scatter(adata, x='n_counts', y='pct_counts_mt', color='dataset_id',
                  ax=axes[1, 1], show=False)
    
    # 小提琴图
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qc_metrics_before_filtering.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"质控图已保存至: {output_dir}/qc_metrics_before_filtering.png")


def filter_cells_and_genes(adata, qc_params):
    """
    根据质控参数过滤细胞和基因
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    qc_params : dict
        质控参数字典
    
    Returns:
    --------
    adata : AnnData
        过滤后的AnnData对象
    """
    print(f"\n{'='*60}")
    print("过滤低质量细胞和基因")
    print(f"{'='*60}")
    
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    print(f"\n过滤前:")
    print(f"  细胞数: {n_cells_before}")
    print(f"  基因数: {n_genes_before}")
    
    # 过滤基因：至少在min_cells个细胞中表达
    sc.pp.filter_genes(adata, min_cells=qc_params['min_cells'])
    
    # 过滤细胞
    adata = adata[
        (adata.obs['n_genes'] >= qc_params['min_genes']) &
        (adata.obs['n_genes'] <= qc_params['max_genes']) &
        (adata.obs['n_counts'] >= qc_params['min_counts']) &
        (adata.obs['pct_counts_mt'] <= qc_params['max_mito_pct'])
    ].copy()
    
    n_cells_after = adata.n_obs
    n_genes_after = adata.n_vars
    
    print(f"\n过滤后:")
    print(f"  细胞数: {n_cells_after} (保留 {n_cells_after/n_cells_before*100:.1f}%)")
    print(f"  基因数: {n_genes_after} (保留 {n_genes_after/n_genes_before*100:.1f}%)")
    
    # 统计每个数据集的细胞数
    print(f"\n各数据集细胞数:")
    for dataset in sorted(adata.obs['dataset_id'].unique()):
        n = (adata.obs['dataset_id'] == dataset).sum()
        print(f"  {dataset}: {n}")
    
    return adata


def normalize_and_scale(adata, n_top_genes=2000):
    """
    归一化和标准化数据
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    n_top_genes : int
        高变基因数量
    
    Returns:
    --------
    adata : AnnData
        处理后的AnnData对象
    """
    print(f"\n{'='*60}")
    print("数据归一化和标准化")
    print(f"{'='*60}")
    
    # 保存原始counts
    adata.layers['counts'] = adata.X.copy()
    
    # 归一化：每个细胞总counts归一化到10000
    print("\n1. 归一化到每个细胞10000 counts...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 对数转换
    print("2. 对数转换...")
    sc.pp.log1p(adata)
    
    # 保存log-normalized数据
    adata.layers['log1p'] = adata.X.copy()
    
    # 识别高变基因
    print(f"3. 识别 {n_top_genes} 个高变基因...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key='dataset_id',  # 考虑批次效应
        flavor='seurat_v3',
        subset=False
    )
    
    print(f"   高变基因数: {adata.var['highly_variable'].sum()}")
    
    # 回归掉技术变异（线粒体百分比和总counts）
    print("4. 回归技术变异...")
    sc.pp.regress_out(adata, ['n_counts', 'pct_counts_mt'])
    
    # 标准化
    print("5. 标准化...")
    sc.pp.scale(adata, max_value=10)
    
    return adata


def batch_correction(adata, batch_key='dataset_id'):
    """
    批次效应校正
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    batch_key : str
        批次信息的列名
    
    Returns:
    --------
    adata : AnnData
        校正后的AnnData对象
    """
    print(f"\n{'='*60}")
    print("批次效应校正 (Harmony)")
    print(f"{'='*60}")
    
    try:
        import scanpy.external as sce
        
        # 先进行PCA
        print("\n1. PCA降维...")
        sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
        
        # 使用Harmony进行批次校正
        print(f"2. Harmony校正 (batch_key={batch_key})...")
        sce.pp.harmony_integrate(
            adata,
            key=batch_key,
            basis='X_pca',
            adjusted_basis='X_pca_harmony'
        )
        
        print("批次校正完成")
        
    except ImportError:
        print("警告: 未安装harmony-pytorch，跳过批次校正")
        print("可以使用: pip install harmonypy")
        
        # 如果没有Harmony，只做PCA
        sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
        adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'].copy()
    
    return adata
