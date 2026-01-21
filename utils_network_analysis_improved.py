"""
H3网络偏倚分析和Normal-Normal差异分析模块（WGCNA版）
使用真正的WGCNA方法进行网络分析
"""
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# 导入WGCNA核心功能
from utils_wgcna import (
    calculate_correlation_matrix,
    calculate_soft_threshold,
    calculate_tom,
    construct_wgcna_network,
    calculate_module_preservation
)


def select_highly_variable_genes(adata_ct, n_genes=500):
    """
    为每个细胞类型选择高变基因
    
    Parameters:
    -----------
    adata_ct : AnnData
        特定细胞类型的数据
    n_genes : int
        选择的基因数量
    
    Returns:
    --------
    hvg_names : list
        高变基因名称列表
    """
    # 获取表达矩阵
    if 'log1p' in adata_ct.layers:
        X = adata_ct.layers['log1p']
    else:
        X = adata_ct.X
    
    if issparse(X):
        X = X.toarray()
    
    # 计算每个基因的方差
    gene_var = np.var(X, axis=0)
    
    # 选择方差最大的基因
    top_idx = np.argsort(gene_var)[-n_genes:]
    hvg_names = adata_ct.var_names[top_idx].tolist()
    
    return hvg_names


def h3_network_bias_improved(adata, output_dir, n_hvgs=500, soft_power=6):
    """
    H3: 网络偏倚分析（WGCNA版）
    使用真正的WGCNA方法进行网络分析
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    n_hvgs : int
        每个细胞类型使用的高变基因数
    soft_power : int
        WGCNA软阈值
    
    Returns:
    --------
    results : dict
        分析结果
    """
    print(f"\n{'='*60}")
    print("H3: 网络偏倚分析（WGCNA方法）")
    print(f"{'='*60}")
    
    results = {}
    all_comparisons = []
    
    # 对每种细胞类型分别分析
    celltypes = [ct for ct in adata.obs['celltype'].unique() if ct != 'Unknown']
    
    for celltype in celltypes:
        print(f"\n{'='*50}")
        print(f"分析细胞类型: {celltype}")
        print(f"{'='*50}")
        
        # 提取该细胞类型的数据
        adata_ct = adata[adata.obs['celltype'] == celltype].copy()
        
        n_cells = adata_ct.n_obs
        datasets = adata_ct.obs['dataset_id'].unique()
        n_datasets = len(datasets)
        
        print(f"  细胞数: {n_cells}")
        print(f"  数据集数: {n_datasets}")
        
        # 检查每个数据集的细胞数
        dataset_sizes = adata_ct.obs['dataset_id'].value_counts()
        print(f"  各数据集细胞数:")
        for ds, count in dataset_sizes.items():
            print(f"    {ds}: {count}")
        
        # 过滤：至少需要2个数据集，每个数据集至少100个细胞
        valid_datasets = [ds for ds in datasets if (adata_ct.obs['dataset_id'] == ds).sum() >= 100]
        
        if len(valid_datasets) < 2:
            print(f"  跳过：有效数据集数不足（需要至少2个，每个≥100细胞）")
            continue
        
        # 1. 选择高变基因
        print(f"\n  1. 选择 {n_hvgs} 个高变基因...")
        hvg_list = select_highly_variable_genes(adata_ct, n_genes=n_hvgs)
        print(f"     实际选择: {len(hvg_list)} 个基因")
        
        # 2. 为每个数据集构建WGCNA网络
        print(f"\n  2. 为每个数据集构建WGCNA网络...")
        
        dataset_modules = {}
        dataset_expr = {}
        
        for dataset in valid_datasets:
            print(f"\n    数据集: {dataset}")
            adata_ds = adata_ct[adata_ct.obs['dataset_id'] == dataset]
            
            # 获取表达矩阵
            if 'log1p' in adata_ds.layers:
                X = adata_ds.layers['log1p']
            else:
                X = adata_ds.X
            
            if issparse(X):
                X = X.toarray()
            
            # 提取高变基因
            gene_indices = [adata_ds.var_names.get_loc(g) for g in hvg_list if g in adata_ds.var_names]
            expr_matrix = X[:, gene_indices]
            actual_genes = [hvg_list[i] for i in range(len(hvg_list)) if hvg_list[i] in adata_ds.var_names]
            
            try:
                # 构建WGCNA网络
                modules, adj_matrix, tom_matrix = construct_wgcna_network(
                    expr_matrix, 
                    actual_genes,
                    soft_power=soft_power,
                    min_module_size=30
                )
                
                dataset_modules[dataset] = modules
                dataset_expr[dataset] = (expr_matrix, actual_genes)
                
                print(f"      识别到 {len(modules)} 个模块")
                
            except Exception as e:
                print(f"      WGCNA失败: {str(e)}")
                continue
        
        if len(dataset_modules) < 2:
            print(f"  跳过：成功构建网络的数据集不足")
            continue
        
        # 3. 比较数据集间的模块保守性
        print(f"\n  3. 比较数据集间的模块保守性...")
        
        comparison_results = []
        
        for i in range(len(valid_datasets)):
            for j in range(i+1, len(valid_datasets)):
                dataset1 = valid_datasets[i]
                dataset2 = valid_datasets[j]
                
                if dataset1 not in dataset_modules or dataset2 not in dataset_modules:
                    continue
                
                print(f"\n    比较: {dataset1} vs {dataset2}")
                
                try:
                    # 计算模块保守性
                    expr1, genes1 = dataset_expr[dataset1]
                    expr2, genes2 = dataset_expr[dataset2]
                    
                    # 找到共同基因
                    common_genes = list(set(genes1) & set(genes2))
                    
                    if len(common_genes) < 50:
                        print(f"      共同基因数不足: {len(common_genes)}")
                        continue
                    
                    # 重新索引表达矩阵
                    idx1 = [genes1.index(g) for g in common_genes]
                    idx2 = [genes2.index(g) for g in common_genes]
                    
                    expr1_common = expr1[:, idx1]
                    expr2_common = expr2[:, idx2]
                    
                    # 计算保守性
                    pres_results = calculate_module_preservation(
                        dataset_modules[dataset1],
                        dataset_modules[dataset2],
                        expr1_common,
                        expr2_common,
                        common_genes
                    )
                    
                    if len(pres_results) > 0:
                        pres_df = pd.DataFrame(pres_results)
                        mean_pres = pres_df['preservation'].mean()
                        
                        print(f"      平均模块保守性: {mean_pres:.3f}")
                        print(f"      比较的模块数: {len(pres_df)}")
                        
                        comparison_results.append({
                            'celltype': celltype,
                            'dataset1': dataset1,
                            'dataset2': dataset2,
                            'n_modules': len(pres_df),
                            'mean_preservation': mean_pres,
                            'n_common_genes': len(common_genes)
                        })
                    
                except Exception as e:
                    print(f"      错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if len(comparison_results) == 0:
            print(f"  该细胞类型无有效比较结果")
            continue
        
        # 4. 汇总统计
        comp_df = pd.DataFrame(comparison_results)
        
        mean_preservation = comp_df['mean_preservation'].mean()
        
        print(f"\n  4. 汇总统计:")
        print(f"     平均模块保守性: {mean_preservation:.3f}")
        print(f"     比较次数: {len(comp_df)}")
        
        # 判断网络偏倚程度
        if mean_preservation > 0.7:
            bias_level = "低"
        elif mean_preservation > 0.5:
            bias_level = "中等"
        else:
            bias_level = "高"
        
        print(f"     网络偏倚程度: {bias_level}")
        
        results[celltype] = {
            'mean_preservation': mean_preservation,
            'n_comparisons': len(comp_df),
            'bias_level': bias_level
        }
        
        all_comparisons.append(comp_df)
        
        # 5. 可视化
        print(f"\n  5. 生成可视化...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 5.1 模块保守性分布
        axes[0].hist(comp_df['mean_preservation'], bins=15, 
                    color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=mean_preservation, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_preservation:.3f}')
        axes[0].axvline(x=0.7, color='green', linestyle=':', 
                       linewidth=2, label='High threshold: 0.7')
        axes[0].axvline(x=0.5, color='orange', linestyle=':', 
                       linewidth=2, label='Medium threshold: 0.5')
        axes[0].set_xlabel('Module Preservation', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{celltype}\nWGCNA Module Preservation', 
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 5.2 模块数量
        axes[1].bar(range(len(comp_df)), comp_df['n_modules'].values,
                   color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Comparison', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Modules', fontsize=11, fontweight='bold')
        axes[1].set_title('Module Counts per Comparison', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/H3_{celltype.replace('/', '_')}_wgcna_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细结果
        comp_df.to_csv(f"{output_dir}/H3_{celltype.replace('/', '_')}_wgcna_comparisons.csv", 
                      index=False)
    
    # 6. 汇总所有细胞类型的结果
    if len(results) > 0:
        print(f"\n{'='*60}")
        print("H3 WGCNA汇总结果")
        print(f"{'='*60}")
        
        summary_df = pd.DataFrame(results).T
        summary_df.to_csv(f"{output_dir}/H3_wgcna_summary.csv")
        
        # 汇总所有比较
        if len(all_comparisons) > 0:
            all_comp_df = pd.concat(all_comparisons, ignore_index=True)
            all_comp_df.to_csv(f"{output_dir}/H3_wgcna_all_comparisons.csv", index=False)
            
            # 总体统计
            overall_preservation = all_comp_df['mean_preservation'].mean()
            
            print(f"\n总体统计:")
            print(f"  平均模块保守性: {overall_preservation:.3f}")
            print(f"  总比较次数: {len(all_comp_df)}")
            
            # 绘制总体汇总图
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 各细胞类型的保守性
            celltype_pres = summary_df['mean_preservation'].sort_values()
            colors = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green' 
                     for x in celltype_pres.values]
            
            axes[0].barh(range(len(celltype_pres)), celltype_pres.values, color=colors, alpha=0.7)
            axes[0].set_yticks(range(len(celltype_pres)))
            axes[0].set_yticklabels(celltype_pres.index, fontsize=10)
            axes[0].axvline(x=0.7, color='green', linestyle='--', label='High (>0.7)')
            axes[0].axvline(x=0.5, color='orange', linestyle='--', label='Medium (>0.5)')
            axes[0].set_xlabel('Mean Module Preservation', fontsize=12, fontweight='bold')
            axes[0].set_title('WGCNA Module Preservation by Cell Type', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(axis='x', alpha=0.3)
            
            # 偏倚程度分布
            bias_counts = summary_df['bias_level'].value_counts()
            colors_pie = {'低': 'green', '中等': 'orange', '高': 'red'}
            pie_colors = [colors_pie.get(x, 'gray') for x in bias_counts.index]
            
            axes[1].pie(bias_counts.values, labels=bias_counts.index, autopct='%1.1f%%',
                       colors=pie_colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            axes[1].set_title('Network Bias Level Distribution', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/H3_wgcna_overall_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 结论
            print(f"\n{'='*60}")
            print("H3 WGCNA结论:")
            if overall_preservation > 0.7:
                print("  ✓ WGCNA模块在数据集间高度保守")
                print("  → 网络偏倚较弱")
            elif overall_preservation > 0.5:
                print("  ⚠ WGCNA模块在数据集间中等保守")
                print("  → 存在一定程度的网络偏倚")
            else:
                print("  ✗ WGCNA模块在数据集间保守性较低")
                print("  → 存在显著的网络偏倚")
            
            print(f"  • {len(summary_df[summary_df['bias_level']=='高'])} 个细胞类型显示高网络偏倚")
            print(f"  • {len(summary_df[summary_df['bias_level']=='中等'])} 个细胞类型显示中等网络偏倚")
            print(f"  • {len(summary_df[summary_df['bias_level']=='低'])} 个细胞类型显示低网络偏倚")
            print(f"{'='*60}")
    
    return results


def normal_normal_differential_analysis(adata, output_dir):
    """
    Normal-Normal差异表达分析
    比较不同数据集的normal样本间的差异表达基因
    
    Parameters:
    -----------
    adata : AnnData
        输入的AnnData对象
    output_dir : str
        输出目录
    
    Returns:
    --------
    results : dict
        差异分析结果
    """
    print(f"\n{'='*60}")
    print("Normal-Normal 差异表达分析")
    print(f"{'='*60}")
    
    results = {}
    all_de_results = []
    
    # 对每种细胞类型分别分析
    celltypes = [ct for ct in adata.obs['celltype'].unique() if ct != 'Unknown']
    
    for celltype in celltypes:
        print(f"\n{'='*50}")
        print(f"分析细胞类型: {celltype}")
        print(f"{'='*50}")
        
        # 提取该细胞类型的数据
        adata_ct = adata[adata.obs['celltype'] == celltype].copy()
        
        n_cells = adata_ct.n_obs
        datasets = sorted(adata_ct.obs['dataset_id'].unique())
        n_datasets = len(datasets)
        
        print(f"  细胞数: {n_cells}")
        print(f"  数据集数: {n_datasets}")
        
        # 检查每个数据集的细胞数
        dataset_sizes = adata_ct.obs['dataset_id'].value_counts()
        print(f"  各数据集细胞数:")
        for ds, count in dataset_sizes.items():
            print(f"    {ds}: {count}")
        
        if n_cells < 100 or n_datasets < 2:
            print(f"  跳过（细胞数或数据集数不足）")
            continue
        
        # 两两比较数据集
        de_results = []
        top_genes_dict = {}
        
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                dataset1 = datasets[i]
                dataset2 = datasets[j]
                
                print(f"\n  比较: {dataset1} vs {dataset2}")
                
                # 提取两个数据集的数据
                adata_d1 = adata_ct[adata_ct.obs['dataset_id'] == dataset1]
                adata_d2 = adata_ct[adata_ct.obs['dataset_id'] == dataset2]
                
                if adata_d1.n_obs < 50 or adata_d2.n_obs < 50:
                    print(f"    细胞数不足（需要≥50），跳过")
                    continue
                
                print(f"    {dataset1}: {adata_d1.n_obs} 细胞")
                print(f"    {dataset2}: {adata_d2.n_obs} 细胞")
                
                try:
                    # 合并两个数据集
                    adata_pair = adata_d1.concatenate(adata_d2, batch_key='comparison_group')
                    adata_pair.obs['comparison_group'] = adata_pair.obs['dataset_id']
                    
                    # 差异表达分析
                    print(f"    执行Wilcoxon秩和检验...")
                    sc.tl.rank_genes_groups(
                        adata_pair,
                        groupby='comparison_group',
                        groups=[dataset1],
                        reference=dataset2,
                        method='wilcoxon',
                        use_raw=False,
                        layer='log1p' if 'log1p' in adata_pair.layers else None,
                        key_added='rank_genes_groups'
                    )
                    
                    # 提取结果
                    de_genes = sc.get.rank_genes_groups_df(adata_pair, group=dataset1, key='rank_genes_groups')
                    
                    # 过滤显著差异基因
                    sig_genes = de_genes[
                        (de_genes['pvals_adj'] < 0.05) & 
                        (np.abs(de_genes['logfoldchanges']) > 0.5)
                    ].copy()
                    
                    n_sig = len(sig_genes)
                    n_up = (sig_genes['logfoldchanges'] > 0).sum()
                    n_down = (sig_genes['logfoldchanges'] < 0).sum()
                    
                    print(f"    显著差异基因: {n_sig} (上调: {n_up}, 下调: {n_down})")
                    
                    # 保存结果
                    de_results.append({
                        'celltype': celltype,
                        'dataset1': dataset1,
                        'dataset2': dataset2,
                        'n_cells_d1': adata_d1.n_obs,
                        'n_cells_d2': adata_d2.n_obs,
                        'n_sig_genes': n_sig,
                        'n_up': n_up,
                        'n_down': n_down
                    })
                    
                    all_de_results.append({
                        'celltype': celltype,
                        'dataset1': dataset1,
                        'dataset2': dataset2,
                        'n_sig_genes': n_sig
                    })
                    
                    # 保存详细的差异基因列表（前100个）
                    if n_sig > 0:
                        top_genes = sig_genes.nlargest(min(100, n_sig), 'scores')
                        comparison_name = f"{dataset1}_vs_{dataset2}"
                        top_genes_dict[comparison_name] = top_genes
                        
                        # 保存完整列表
                        sig_genes.to_csv(
                            f"{output_dir}/DE_{celltype.replace('/', '_')}_{dataset1}_vs_{dataset2}.csv",
                            index=False
                        )
                    
                except Exception as e:
                    print(f"    差异分析失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if len(de_results) > 0:
            results[celltype] = {
                'de_summary': pd.DataFrame(de_results),
                'top_genes': top_genes_dict
            }
            
            # 可视化该细胞类型的结果
            print(f"\n  生成可视化...")
            
            de_df = pd.DataFrame(de_results)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 差异基因数柱状图
            comparison_labels = [f"{row['dataset1']}\nvs\n{row['dataset2']}" 
                               for _, row in de_df.iterrows()]
            
            x_pos = np.arange(len(de_df))
            axes[0].bar(x_pos, de_df['n_sig_genes'], color='steelblue', alpha=0.7, edgecolor='black')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(comparison_labels, fontsize=8, rotation=45, ha='right')
            axes[0].set_ylabel('Number of DEGs', fontsize=11, fontweight='bold')
            axes[0].set_title(f'{celltype}\nDifferential Gene Counts', fontsize=12, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
            
            # 上调/下调基因堆叠图
            axes[1].bar(x_pos, de_df['n_up'], label='Upregulated', 
                       color='red', alpha=0.7, edgecolor='black')
            axes[1].bar(x_pos, de_df['n_down'], bottom=de_df['n_up'], 
                       label='Downregulated', color='blue', alpha=0.7, edgecolor='black')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(comparison_labels, fontsize=8, rotation=45, ha='right')
            axes[1].set_ylabel('Number of DEGs', fontsize=11, fontweight='bold')
            axes[1].set_title(f'{celltype}\nUp/Down-regulated Genes', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/DE_{celltype.replace('/', '_')}_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 汇总所有细胞类型的结果
    if len(all_de_results) > 0:
        print(f"\n{'='*60}")
        print("Normal-Normal 差异分析汇总")
        print(f"{'='*60}")
        
        all_de_df = pd.DataFrame(all_de_results)
        all_de_df.to_csv(f"{output_dir}/Normal_Normal_DE_summary.csv", index=False)
        
        # 统计
        mean_degs = all_de_df['n_sig_genes'].mean()
        median_degs = all_de_df['n_sig_genes'].median()
        max_degs = all_de_df['n_sig_genes'].max()
        min_degs = all_de_df['n_sig_genes'].min()
        
        print(f"\n总体统计:")
        print(f"  总比较次数: {len(all_de_df)}")
        print(f"  平均差异基因数: {mean_degs:.1f}")
        print(f"  中位数差异基因数: {median_degs:.1f}")
        print(f"  最大差异基因数: {max_degs}")
        print(f"  最小差异基因数: {min_degs}")
        
        # 各细胞类型的平均差异基因数
        celltype_avg = all_de_df.groupby('celltype')['n_sig_genes'].agg(['mean', 'std', 'count'])
        celltype_avg = celltype_avg.sort_values('mean', ascending=False)
        
        print(f"\n各细胞类型的平均差异基因数:")
        for celltype, row in celltype_avg.iterrows():
            print(f"  {celltype}: {row['mean']:.1f} ± {row['std']:.1f} (n={int(row['count'])})")
        
        # 可视化汇总
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 各细胞类型的平均差异基因数
        axes[0, 0].barh(range(len(celltype_avg)), celltype_avg['mean'].values, 
                       xerr=celltype_avg['std'].values,
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_yticks(range(len(celltype_avg)))
        axes[0, 0].set_yticklabels(celltype_avg.index, fontsize=10)
        axes[0, 0].set_xlabel('Mean Number of DEGs', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Average DEGs by Cell Type', fontsize=12, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. 差异基因数分布
        axes[0, 1].hist(all_de_df['n_sig_genes'], bins=20, 
                       color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=mean_degs, color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {mean_degs:.1f}')
        axes[0, 1].axvline(x=median_degs, color='blue', linestyle='--', 
                          linewidth=2, label=f'Median: {median_degs:.1f}')
        axes[0, 1].set_xlabel('Number of DEGs', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Distribution of DEG Counts', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. 热图：各细胞类型在不同比较中的差异基因数
        pivot_data = all_de_df.pivot_table(
            values='n_sig_genes',
            index='celltype',
            columns=['dataset1', 'dataset2'],
            aggfunc='mean'
        )
        
        if pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd',
                       cbar_kws={'label': 'Number of DEGs'},
                       ax=axes[1, 0], linewidths=0.5)
            axes[1, 0].set_xlabel('Dataset Comparison', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Cell Type', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('DEGs Heatmap', fontsize=12, fontweight='bold')
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(axes[1, 0].get_yticklabels(), rotation=0, fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for heatmap', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
        
        # 4. 箱线图：各细胞类型的差异基因数分布
        celltype_order = celltype_avg.index.tolist()
        plot_data = all_de_df[all_de_df['celltype'].isin(celltype_order)]
        
        if len(plot_data) > 0:
            box_data = [plot_data[plot_data['celltype'] == ct]['n_sig_genes'].values 
                       for ct in celltype_order]
            
            bp = axes[1, 1].boxplot(box_data, labels=celltype_order, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            axes[1, 1].set_xticklabels(celltype_order, rotation=45, ha='right', fontsize=9)
            axes[1, 1].set_ylabel('Number of DEGs', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('DEG Distribution by Cell Type', fontsize=12, fontweight='bold')
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Normal_Normal_DE_overall_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 结论
        print(f"\n{'='*60}")
        print("Normal-Normal 差异分析结论:")
        
        if mean_degs > 200:
            print("  ✗ Normal样本间存在大量差异表达基因")
            print("  → 提示显著的样本间异质性")
        elif mean_degs > 100:
            print("  ⚠ Normal样本间存在中等数量的差异表达基因")
            print("  → 提示一定程度的样本间异质性")
        else:
            print("  ✓ Normal样本间差异表达基因较少")
            print("  → 样本间异质性较低")
        
        # 识别异质性最高的细胞类型
        top3_heterogeneous = celltype_avg.head(3)
        print(f"\n  异质性最高的细胞类型:")
        for idx, (celltype, row) in enumerate(top3_heterogeneous.iterrows(), 1):
            print(f"    {idx}. {celltype}: 平均 {row['mean']:.1f} 个DEGs")
        
        print(f"{'='*60}")
    
    return results