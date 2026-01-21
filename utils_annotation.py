"""
细胞类型注释模块
采用多层次、统计学严谨的注释策略
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, mannwhitneyu, ranksums
from scipy.sparse import issparse


def clustering(adata, resolution=0.8, use_harmony=True):
    """
    聚类分析
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    resolution : float
        聚类分辨率
    use_harmony : bool
        是否使用Harmony校正后的PCA
    
    Returns:
    --------
    adata : AnnData
        添加了聚类结果的AnnData对象
    """
    print(f"\n{'='*60}")
    print("聚类分析")
    print(f"{'='*60}")
    
    # 选择使用的PCA结果
    pca_key = 'X_pca_harmony' if use_harmony and 'X_pca_harmony' in adata.obsm else 'X_pca'
    print(f"\n使用 {pca_key} 进行聚类")
    
    # 计算邻居图
    print("1. 计算邻居图...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep=pca_key)
    
    # UMAP降维
    print("2. UMAP降维...")
    sc.tl.umap(adata)
    
    # Leiden聚类
    print(f"3. Leiden聚类 (resolution={resolution})...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    
    n_clusters = adata.obs['leiden'].nunique()
    print(f"   识别到 {n_clusters} 个聚类")
    
    # 统计每个聚类的细胞数
    print("\n各聚类细胞数:")
    cluster_counts = adata.obs['leiden'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count}")
    
    return adata


def calculate_marker_scores(adata, marker_dict):
    """
    计算每个细胞对各细胞类型marker基因的表达评分
    使用标准化的方法，避免偏倚
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    marker_dict : dict
        细胞类型到marker基因列表的字典
    
    Returns:
    --------
    adata : AnnData
        添加了marker评分的AnnData对象
    marker_scores : DataFrame
        原始marker评分矩阵
    marker_scores_zscore : DataFrame
        Z-score标准化后的评分矩阵
    """
    print(f"\n{'='*60}")
    print("计算marker基因评分")
    print(f"{'='*60}")
    
    # 使用log-normalized数据
    if 'log1p' in adata.layers:
        X = adata.layers['log1p']
    else:
        X = adata.X
    
    # 转换为dense array（如果是sparse）
    if issparse(X):
        X = X.toarray()
    
    marker_scores = pd.DataFrame(index=adata.obs_names)
    marker_info = {}
    
    for cell_type, markers in marker_dict.items():
        # 找到在数据中存在的marker基因
        available_markers = [m for m in markers if m in adata.var_names]
        
        if len(available_markers) == 0:
            print(f"  {cell_type}: 无可用marker基因")
            marker_scores[cell_type] = 0
            marker_info[cell_type] = {'n_markers': 0, 'markers': []}
            continue
        
        # 获取marker基因的索引
        marker_indices = [adata.var_names.get_loc(m) for m in available_markers]
        
        # 计算平均表达
        marker_expr = X[:, marker_indices].mean(axis=1)
        
        # 如果是matrix，转换为1D array
        if hasattr(marker_expr, 'A1'):
            marker_expr = marker_expr.A1
        elif len(marker_expr.shape) > 1:
            marker_expr = marker_expr.flatten()
        
        marker_scores[cell_type] = marker_expr
        marker_info[cell_type] = {
            'n_markers': len(available_markers),
            'markers': available_markers
        }
        
        print(f"  {cell_type}: {len(available_markers)}/{len(markers)} 个marker基因可用")
    
    # Z-score标准化：使每个细胞类型的评分具有可比性
    print("\n标准化marker评分（Z-score）...")
    marker_scores_zscore = marker_scores.apply(zscore, axis=0)
    
    # 保存到adata
    for col in marker_scores.columns:
        adata.obs[f'score_{col}'] = marker_scores[col].values
        adata.obs[f'score_zscore_{col}'] = marker_scores_zscore[col].values
    
    return adata, marker_scores, marker_scores_zscore, marker_info


def calculate_cluster_specificity(adata, marker_scores_zscore, marker_info):
    """
    计算每个聚类对各细胞类型的特异性评分
    使用相对评分和统计检验
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    marker_scores_zscore : DataFrame
        Z-score标准化后的marker评分
    marker_info : dict
        marker基因信息
    
    Returns:
    --------
    cluster_specificity : DataFrame
        聚类特异性评分矩阵
    """
    print(f"\n计算聚类特异性评分...")
    
    clusters = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))
    celltypes = marker_scores_zscore.columns
    
    # 存储结果
    specificity_scores = []
    mean_scores = []
    median_scores = []
    pct_positive = []
    
    for cluster in clusters:
        mask = adata.obs['leiden'] == cluster
        
        # 该聚类的评分
        cluster_scores = marker_scores_zscore.loc[mask]
        
        # 其他聚类的评分
        other_scores = marker_scores_zscore.loc[~mask]
        
        spec_row = {}
        mean_row = {}
        median_row = {}
        pct_row = {}
        
        for celltype in celltypes:
            if marker_info[celltype]['n_markers'] == 0:
                spec_row[celltype] = 0
                mean_row[celltype] = 0
                median_row[celltype] = 0
                pct_row[celltype] = 0
                continue
            
            # 平均Z-score
            mean_z = cluster_scores[celltype].mean()
            median_z = cluster_scores[celltype].median()
            
            # 阳性细胞比例（Z-score > 0.5）
            pct_pos = (cluster_scores[celltype] > 0.5).sum() / len(cluster_scores) * 100
            
            # 相对特异性：该聚类的平均评分 vs 其他聚类的平均评分
            other_mean_z = other_scores[celltype].mean()
            relative_specificity = mean_z - other_mean_z
            
            spec_row[celltype] = relative_specificity
            mean_row[celltype] = mean_z
            median_row[celltype] = median_z
            pct_row[celltype] = pct_pos
        
        specificity_scores.append(spec_row)
        mean_scores.append(mean_row)
        median_scores.append(median_row)
        pct_positive.append(pct_row)
    
    # 转换为DataFrame
    cluster_specificity = pd.DataFrame(specificity_scores, index=[f'Cluster_{c}' for c in clusters])
    cluster_mean = pd.DataFrame(mean_scores, index=[f'Cluster_{c}' for c in clusters])
    cluster_median = pd.DataFrame(median_scores, index=[f'Cluster_{c}' for c in clusters])
    cluster_pct = pd.DataFrame(pct_positive, index=[f'Cluster_{c}' for c in clusters])
    
    return cluster_specificity, cluster_mean, cluster_median, cluster_pct


def annotate_clusters(adata, marker_scores, marker_scores_zscore, marker_info, 
                      min_cluster_size=50, min_specificity=0.3, min_mean_zscore=0.5):
    """
    基于多重标准注释聚类
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    marker_scores : DataFrame
        原始marker评分
    marker_scores_zscore : DataFrame
        Z-score标准化后的评分
    marker_info : dict
        marker基因信息
    min_cluster_size : int
        最小聚类大小
    min_specificity : float
        最小特异性阈值
    min_mean_zscore : float
        最小平均Z-score阈值
    
    Returns:
    --------
    adata : AnnData
        添加了细胞类型注释的AnnData对象
    """
    print(f"\n{'='*60}")
    print("注释细胞类型（多重标准）")
    print(f"{'='*60}")
    
    # 计算聚类特异性
    cluster_specificity, cluster_mean, cluster_median, cluster_pct = calculate_cluster_specificity(
        adata, marker_scores_zscore, marker_info
    )
    
    # 注释策略
    cluster_annotations = {}
    annotation_confidence = {}
    
    print(f"\n{'聚类':<12} {'细胞数':<10} {'注释':<35} {'特异性':<10} {'平均Z':<10} {'阳性%':<10} {'置信度':<10}")
    print("-" * 110)
    
    for cluster in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
        mask = adata.obs['leiden'] == cluster
        cluster_size = mask.sum()
        
        # 过滤小聚类
        if cluster_size < min_cluster_size:
            cluster_annotations[cluster] = 'Remove'
            annotation_confidence[cluster] = 0
            print(f"Cluster {cluster:<5} {cluster_size:<10} {'Remove (too small)':<35} {'-':<10} {'-':<10} {'-':<10} {'Low':<10}")
            continue
        
        cluster_name = f'Cluster_{cluster}'
        
        # 获取该聚类的评分
        spec_scores = cluster_specificity.loc[cluster_name]
        mean_scores = cluster_mean.loc[cluster_name]
        pct_scores = cluster_pct.loc[cluster_name]
        
        # 筛选候选细胞类型：特异性和平均Z-score都要达标
        candidates = []
        for celltype in spec_scores.index:
            if marker_info[celltype]['n_markers'] == 0:
                continue
            
            spec = spec_scores[celltype]
            mean_z = mean_scores[celltype]
            pct = pct_scores[celltype]
            
            # 综合评分：特异性 + 平均Z-score + 阳性比例
            combined_score = spec * 0.4 + mean_z * 0.4 + (pct / 100) * 0.2
            
            if spec > min_specificity and mean_z > min_mean_zscore:
                candidates.append({
                    'celltype': celltype,
                    'specificity': spec,
                    'mean_zscore': mean_z,
                    'pct_positive': pct,
                    'combined_score': combined_score
                })
        
        # 选择最佳候选
        if len(candidates) == 0:
            # 没有明确的候选，标记为Unknown
            cluster_annotations[cluster] = 'Unknown'
            annotation_confidence[cluster] = 0
            
            # 但仍然显示最高评分的类型作为参考
            best_celltype = spec_scores.idxmax()
            best_spec = spec_scores[best_celltype]
            best_mean = mean_scores[best_celltype]
            best_pct = pct_scores[best_celltype]
            
            print(f"Cluster {cluster:<5} {cluster_size:<10} {'Unknown (' + best_celltype + '?)':<35} "
                  f"{best_spec:<10.3f} {best_mean:<10.3f} {best_pct:<10.1f} {'Low':<10}")
        else:
            # 按综合评分排序
            candidates_df = pd.DataFrame(candidates).sort_values('combined_score', ascending=False)
            best = candidates_df.iloc[0]
            
            # 计算置信度
            if len(candidates_df) > 1:
                second_best = candidates_df.iloc[1]
                score_gap = best['combined_score'] - second_best['combined_score']
                confidence = min(1.0, score_gap / 0.5)  # 归一化到0-1
            else:
                confidence = 1.0
            
            # 根据置信度决定是否添加"?"
            if confidence < 0.3:
                celltype_label = f"{best['celltype']}?"
                conf_label = 'Low'
            elif confidence < 0.6:
                celltype_label = best['celltype']
                conf_label = 'Medium'
            else:
                celltype_label = best['celltype']
                conf_label = 'High'
            
            cluster_annotations[cluster] = best['celltype']
            annotation_confidence[cluster] = confidence
            
            print(f"Cluster {cluster:<5} {cluster_size:<10} {celltype_label:<35} "
                  f"{best['specificity']:<10.3f} {best['mean_zscore']:<10.3f} "
                  f"{best['pct_positive']:<10.1f} {conf_label:<10}")
    
    # 添加注释到adata
    adata.obs['celltype'] = adata.obs['leiden'].map(cluster_annotations)
    adata.obs['annotation_confidence'] = adata.obs['leiden'].map(annotation_confidence)
    
    # 过滤掉标记为删除的细胞
    n_before = adata.n_obs
    adata = adata[adata.obs['celltype'] != 'Remove'].copy()
    n_after = adata.n_obs
    
    print(f"\n过滤小亚群后:")
    print(f"  细胞数: {n_before} -> {n_after} (移除 {n_before - n_after})")
    
    # 统计每种细胞类型的数量
    print(f"\n各细胞类型细胞数:")
    celltype_counts = adata.obs['celltype'].value_counts()
    for celltype, count in celltype_counts.items():
        print(f"  {celltype}: {count}")
    
    # 保存注释详情
    annotation_details = pd.DataFrame({
        'cluster': list(cluster_annotations.keys()),
        'celltype': list(cluster_annotations.values()),
        'confidence': list(annotation_confidence.values())
    })
    
    return adata, annotation_details, cluster_specificity, cluster_mean


def plot_annotation_results(adata, output_dir, cluster_specificity=None, cluster_mean=None):
    """
    绘制注释结果（增强版）
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    cluster_specificity : DataFrame, optional
        聚类特异性矩阵
    cluster_mean : DataFrame, optional
        聚类平均评分矩阵
    """
    print(f"\n绘制注释结果...")
    
    # 设置图形样式
    sc.set_figure_params(dpi=100, frameon=False, figsize=(8, 8))
    
    # 1. UMAP by celltype
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata, color='celltype', ax=ax, show=False, legend_loc='right margin',
               title='Cell Type Annotation', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_celltype.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP by dataset
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata, color='dataset_id', ax=ax, show=False, legend_loc='right margin',
               title='Dataset Distribution', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. UMAP by disease
    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.umap(adata, color='disease', ax=ax, show=False, legend_loc='right margin',
               title='Disease Type', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_disease.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. UMAP by annotation confidence
    if 'annotation_confidence' in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        sc.pl.umap(adata, color='annotation_confidence', ax=ax, show=False,
                   title='Annotation Confidence', frameon=False, cmap='RdYlGn')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_confidence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 细胞类型组成堆叠柱状图
    fig, ax = plt.subplots(figsize=(14, 7))
    
    composition = pd.crosstab(
        adata.obs['dataset_id'],
        adata.obs['celltype'],
        normalize='index'
    ) * 100
    
    composition.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Composition by Dataset', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/composition_by_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 聚类特异性热图
    if cluster_specificity is not None:
        fig, ax = plt.subplots(figsize=(14, max(8, len(cluster_specificity) * 0.4)))
        
        # 只显示有marker的细胞类型
        valid_cols = [col for col in cluster_specificity.columns 
                     if cluster_specificity[col].abs().max() > 0]
        
        if len(valid_cols) > 0:
            plot_data = cluster_specificity[valid_cols]
            
            sns.heatmap(
                plot_data,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Specificity Score'},
                ax=ax,
                linewidths=0.5
            )
            ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
            ax.set_title('Cluster Specificity Scores', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cluster_specificity_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 7. 聚类平均评分热图
    if cluster_mean is not None:
        fig, ax = plt.subplots(figsize=(14, max(8, len(cluster_mean) * 0.4)))
        
        # 只显示有marker的细胞类型
        valid_cols = [col for col in cluster_mean.columns 
                     if cluster_mean[col].abs().max() > 0]
        
        if len(valid_cols) > 0:
            plot_data = cluster_mean[valid_cols]
            
            sns.heatmap(
                plot_data,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Mean Z-score'},
                ax=ax,
                linewidths=0.5
            )
            ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
            ax.set_title('Cluster Mean Marker Scores (Z-score)', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cluster_mean_scores_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 8. 细胞类型比例对比（各数据集）
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 绝对数量
    composition_abs = pd.crosstab(
        adata.obs['dataset_id'],
        adata.obs['celltype']
    )
    
    composition_abs.plot(kind='bar', ax=axes[0], colormap='tab20')
    axes[0].set_xlabel('Dataset', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Cells', fontsize=11, fontweight='bold')
    axes[0].set_title('Cell Type Counts by Dataset', fontsize=13, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 相对比例
    composition.plot(kind='bar', ax=axes[1], colormap='tab20')
    axes[1].set_xlabel('Dataset', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cell Type Proportions by Dataset', fontsize=13, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/composition_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"注释结果图已保存至: {output_dir}/")
