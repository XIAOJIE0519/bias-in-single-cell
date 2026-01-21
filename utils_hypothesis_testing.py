"""
假设检验分析模块：H1组成偏倚、H2状态偏倚、H3网络偏倚
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def h1_composition_bias(adata, output_dir):
    """
    H1: 组成偏倚分析
    检验normal样本的细胞组成在不同研究间是否存在显著差异
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    
    Returns:
    --------
    results : dict
        分析结果
    """
    print(f"\n{'='*60}")
    print("H1: 组成偏倚分析")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. 计算每个样本的细胞类型组成
    print("\n1. 计算细胞类型组成...")
    
    composition = pd.crosstab(
        [adata.obs['dataset_id'], adata.obs['sample_id']],
        adata.obs['celltype']
    )
    
    # 转换为比例
    composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100
    
    # 添加数据集信息
    composition_pct['dataset_id'] = [idx[0] for idx in composition_pct.index]
    
    print(f"   样本数: {len(composition_pct)}")
    print(f"   细胞类型数: {len(composition.columns)}")
    
    # 2. 卡方检验：检验细胞类型分布是否独立于数据集
    print("\n2. 卡方检验...")
    
    # 按数据集汇总
    composition_by_dataset = pd.crosstab(
        adata.obs['dataset_id'],
        adata.obs['celltype']
    )
    
    chi2, pval, dof, expected = stats.chi2_contingency(composition_by_dataset)
    
    print(f"   Chi-square statistic: {chi2:.2f}")
    print(f"   P-value: {pval:.2e}")
    print(f"   Degrees of freedom: {dof}")
    
    results['chi2_test'] = {
        'chi2': chi2,
        'pvalue': pval,
        'dof': dof,
        'significant': pval < 0.05
    }
    
    # 3. 计算Jensen-Shannon散度（衡量分布差异）
    print("\n3. 计算数据集间组成差异 (Jensen-Shannon Divergence)...")
    
    composition_by_dataset_pct = composition_by_dataset.div(
        composition_by_dataset.sum(axis=1), axis=0
    )
    
    js_distances = []
    dataset_pairs = []
    
    datasets = composition_by_dataset_pct.index.tolist()
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            p = composition_by_dataset_pct.iloc[i].values
            q = composition_by_dataset_pct.iloc[j].values
            
            # Jensen-Shannon散度
            m = (p + q) / 2
            js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
            
            js_distances.append(js_div)
            dataset_pairs.append(f"{datasets[i]} vs {datasets[j]}")
    
    js_df = pd.DataFrame({
        'pair': dataset_pairs,
        'js_divergence': js_distances
    })
    
    print(f"\n   平均JS散度: {np.mean(js_distances):.4f}")
    print(f"   最大JS散度: {np.max(js_distances):.4f}")
    print(f"   最小JS散度: {np.min(js_distances):.4f}")
    
    results['js_divergence'] = js_df
    
    # 4. 方差分析：每种细胞类型在数据集间的比例差异
    print("\n4. 各细胞类型的数据集间差异 (Kruskal-Wallis检验)...")
    
    celltype_anova = []
    
    for celltype in composition.columns:
        # 提取每个数据集的该细胞类型比例
        groups = []
        for dataset in composition_pct['dataset_id'].unique():
            mask = composition_pct['dataset_id'] == dataset
            groups.append(composition_pct.loc[mask, celltype].values)
        
        # Kruskal-Wallis检验（非参数）
        if len(groups) > 1:
            h_stat, p_val = stats.kruskal(*groups)
            
            celltype_anova.append({
                'celltype': celltype,
                'h_statistic': h_stat,
                'pvalue': p_val,
                'mean_pct': composition_pct[celltype].mean(),
                'std_pct': composition_pct[celltype].std()
            })
    
    celltype_anova_df = pd.DataFrame(celltype_anova).sort_values('pvalue')
    
    # FDR校正
    from statsmodels.stats.multitest import multipletests
    _, celltype_anova_df['pvalue_adj'], _, _ = multipletests(
        celltype_anova_df['pvalue'],
        method='fdr_bh'
    )
    
    print(f"\n   显著差异的细胞类型 (FDR < 0.05):")
    sig_celltypes = celltype_anova_df[celltype_anova_df['pvalue_adj'] < 0.05]
    for _, row in sig_celltypes.iterrows():
        print(f"     {row['celltype']}: p_adj = {row['pvalue_adj']:.2e}")
    
    results['celltype_variance'] = celltype_anova_df
    
    # 5. 可视化
    print("\n5. 生成可视化图表...")
    
    # 5.1 堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    composition_by_dataset_pct.T.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='tab20'
    )
    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('H1: Cell Type Composition by Dataset', fontsize=14, fontweight='bold')
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H1_composition_stacked_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5.2 热图
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        composition_by_dataset_pct,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('H1: Cell Type Composition Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H1_composition_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5.3 箱线图：显示显著差异的细胞类型
    if len(sig_celltypes) > 0:
        n_sig = min(6, len(sig_celltypes))
        top_sig = sig_celltypes.head(n_sig)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_sig.iterrows()):
            if idx >= 6:
                break
            
            celltype = row['celltype']
            
            # 准备数据
            plot_data = composition_pct[['dataset_id', celltype]].copy()
            plot_data.columns = ['Dataset', 'Percentage']
            
            sns.boxplot(
                data=plot_data,
                x='Dataset',
                y='Percentage',
                ax=axes[idx],
                palette='Set2'
            )
            sns.swarmplot(
                data=plot_data,
                x='Dataset',
                y='Percentage',
                ax=axes[idx],
                color='black',
                alpha=0.5,
                size=3
            )
            
            axes[idx].set_title(f"{celltype}\n(p_adj = {row['pvalue_adj']:.2e})", 
                               fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for idx in range(len(top_sig), 6):
            axes[idx].axis('off')
        
        plt.suptitle('H1: Cell Types with Significant Composition Differences', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/H1_composition_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存结果
    composition_pct.to_csv(f"{output_dir}/H1_composition_table.csv")
    celltype_anova_df.to_csv(f"{output_dir}/H1_celltype_variance_test.csv", index=False)
    js_df.to_csv(f"{output_dir}/H1_js_divergence.csv", index=False)
    
    print(f"\n结果已保存至: {output_dir}/")
    
    # 结论
    print(f"\n{'='*60}")
    print("H1 结论:")
    if results['chi2_test']['significant']:
        print("  ✓ 细胞组成在数据集间存在显著差异 (p < 0.05)")
        print(f"  ✓ {len(sig_celltypes)}/{len(celltype_anova_df)} 种细胞类型显示显著差异")
        print("  → 存在组成偏倚")
    else:
        print("  ✗ 细胞组成在数据集间无显著差异")
        print("  → 不存在组成偏倚")
    print(f"{'='*60}")
    
    return results


def h2_state_bias(adata, output_dir, n_hvgs=2000):
    """
    H2: 状态偏倚分析
    检验同一细胞类型内，normal样本是否仍显著分化，study是否解释大量方差
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    n_hvgs : int
        用于分析的高变基因数量
    
    Returns:
    --------
    results : dict
        分析结果
    """
    print(f"\n{'='*60}")
    print("H2: 状态偏倚分析")
    print(f"{'='*60}")
    
    results = {}
    
    # 对每种细胞类型分别分析
    celltypes = adata.obs['celltype'].unique()
    
    for celltype in celltypes:
        print(f"\n{'='*50}")
        print(f"分析细胞类型: {celltype}")
        print(f"{'='*50}")
        
        # 提取该细胞类型的数据
        adata_ct = adata[adata.obs['celltype'] == celltype].copy()
        
        n_cells = adata_ct.n_obs
        n_datasets = adata_ct.obs['dataset_id'].nunique()
        
        print(f"  细胞数: {n_cells}")
        print(f"  数据集数: {n_datasets}")
        
        if n_cells < 100 or n_datasets < 2:
            print(f"  跳过（细胞数或数据集数不足）")
            continue
        
        # 1. PCA分析
        print("\n  1. PCA降维...")
        
        # 使用log-normalized数据
        if 'log1p' in adata_ct.layers:
            X = adata_ct.layers['log1p'].toarray() if hasattr(adata_ct.layers['log1p'], 'toarray') else adata_ct.layers['log1p']
        else:
            X = adata_ct.X.toarray() if hasattr(adata_ct.X, 'toarray') else adata_ct.X
        
        # 选择高变基因
        gene_var = np.var(X, axis=0)
        top_genes_idx = np.argsort(gene_var)[-n_hvgs:]
        X_hvg = X[:, top_genes_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_hvg)
        
        # PCA
        pca = PCA(n_components=min(50, n_cells-1))
        X_pca = pca.fit_transform(X_scaled)
        
        # 保存PCA结果
        adata_ct.obsm['X_pca_ct'] = X_pca
        
        print(f"     前10个PC解释的方差: {pca.explained_variance_ratio_[:10].sum():.2%}")
        
        # 2. PERMANOVA分析：检验数据集是否解释显著方差
        print("\n  2. PERMANOVA分析...")
        
        try:
            from skbio.stats.distance import permanova
            from scipy.spatial.distance import pdist, squareform
            
            # 计算欧氏距离矩阵
            dist_matrix = squareform(pdist(X_pca[:, :30], metric='euclidean'))
            
            # PERMANOVA检验
            perm_result = permanova(
                dist_matrix,
                adata_ct.obs['dataset_id'],
                permutations=999
            )
            
            print(f"     F-statistic: {perm_result['test statistic']:.4f}")
            print(f"     P-value: {perm_result['p-value']:.4f}")
            print(f"     R²: {perm_result['test statistic'] / (perm_result['test statistic'] + perm_result['number of groups'] - 1):.4f}")
            
            results[celltype] = {
                'permanova_f': perm_result['test statistic'],
                'permanova_p': perm_result['p-value'],
                'n_cells': n_cells,
                'n_datasets': n_datasets
            }
            
        except ImportError:
            print("     警告: 未安装scikit-bio，使用替代方法")
            
            # 使用方差分析作为替代
            from sklearn.metrics import silhouette_score
            
            # 计算轮廓系数
            if len(np.unique(adata_ct.obs['dataset_id'])) > 1:
                sil_score = silhouette_score(X_pca[:, :30], adata_ct.obs['dataset_id'])
                print(f"     Silhouette score: {sil_score:.4f}")
                
                results[celltype] = {
                    'silhouette_score': sil_score,
                    'n_cells': n_cells,
                    'n_datasets': n_datasets
                }
        
        # 3. 方差分解：计算数据集解释的方差比例
        print("\n  3. 方差分解分析...")
        
        # 对每个PC，计算数据集间方差占总方差的比例
        variance_explained = []
        
        for pc_idx in range(min(20, X_pca.shape[1])):
            pc_values = X_pca[:, pc_idx]
            
            # 总方差
            total_var = np.var(pc_values)
            
            # 数据集间方差
            dataset_means = adata_ct.obs.groupby('dataset_id').apply(
                lambda x: pc_values[x.index.get_indexer(adata_ct.obs_names)].mean()
            )
            
            between_var = 0
            for dataset in adata_ct.obs['dataset_id'].unique():
                mask = adata_ct.obs['dataset_id'] == dataset
                n_dataset = mask.sum()
                dataset_mean = pc_values[mask].mean()
                overall_mean = pc_values.mean()
                between_var += n_dataset * (dataset_mean - overall_mean) ** 2
            
            between_var /= len(pc_values)
            
            var_ratio = between_var / total_var if total_var > 0 else 0
            variance_explained.append(var_ratio)
        
        mean_var_explained = np.mean(variance_explained)
        print(f"     数据集平均解释方差比例: {mean_var_explained:.2%}")
        
        if celltype in results:
            results[celltype]['variance_explained_by_dataset'] = mean_var_explained
        
        # 4. 可视化
        print("\n  4. 生成可视化...")
        
        # 4.1 PCA图（按数据集着色）
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        datasets = adata_ct.obs['dataset_id'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        
        for idx, dataset in enumerate(datasets):
            mask = adata_ct.obs['dataset_id'] == dataset
            axes[0].scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[colors[idx]],
                label=dataset,
                alpha=0.6,
                s=10
            )
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0].set_title(f'{celltype} - PCA by Dataset')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 4.2 方差解释图
        axes[1].bar(range(1, len(variance_explained)+1), variance_explained, color='steelblue')
        axes[1].axhline(y=mean_var_explained, color='red', linestyle='--', 
                       label=f'Mean: {mean_var_explained:.2%}')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Variance Explained by Dataset')
        axes[1].set_title(f'{celltype} - Dataset Effect on PCs')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/H2_{celltype.replace('/', '_')}_state_bias.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 汇总结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{output_dir}/H2_state_bias_summary.csv")
    
    print(f"\n{'='*60}")
    print("H2 结论:")
    
    if len(results_df) > 0:
        # 统计显著的细胞类型
        if 'permanova_p' in results_df.columns:
            sig_celltypes = results_df[results_df['permanova_p'] < 0.05]
            print(f"  ✓ {len(sig_celltypes)}/{len(results_df)} 种细胞类型显示显著的数据集效应")
        
        if 'variance_explained_by_dataset' in results_df.columns:
            mean_var = results_df['variance_explained_by_dataset'].mean()
            print(f"  ✓ 数据集平均解释 {mean_var:.1%} 的方差")
            
            if mean_var > 0.1:
                print("  → 存在显著的状态偏倚")
            else:
                print("  → 状态偏倚较弱")
    
    print(f"{'='*60}")
    
    return results
