"""
主分析脚本：完整的单细胞数据分析流程
按照Nature顶刊标准进行严谨的分析
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from config import *
from utils_data_loading import load_all_datasets, load_marker_genes
from utils_preprocessing import (
    calculate_qc_metrics,
    plot_qc_metrics,
    filter_cells_and_genes,
    normalize_and_scale,
    batch_correction
)
from utils_annotation import (
    clustering,
    calculate_marker_scores,
    annotate_clusters,
    plot_annotation_results
)
from utils_hypothesis_testing import (
    h1_composition_bias,
    h2_state_bias
)
from utils_network_analysis_improved import (
    h3_network_bias_improved,
    normal_normal_differential_analysis
)
from utils_result_tables import generate_all_result_tables

# 设置随机种子
np.random.seed(RANDOM_SEED)

# 设置scanpy参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sc.settings.figdir = FIGURE_DIR

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║           单细胞RNA-seq数据分析：入院率偏倚研究                              ║
║                                                                           ║
║           Single-cell RNA-seq Analysis: Admission Rate Bias Study        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

def main():
    """
    主分析流程
    """
    
    # ========================================================================
    # 第一部分：数据加载
    # ========================================================================
    print("\n" + "="*80)
    print("第一部分：数据加载")
    print("="*80)
    
    # 1.1 加载所有数据集
    print("\n1.1 加载所有数据集...")
    adata = load_all_datasets(DATASETS, BASE_DIR)
    
    # 1.2 加载marker基因
    print("\n1.2 加载marker基因注释...")
    marker_dict = load_marker_genes(MARKER_FILE)
    
    # 保存原始数据信息
    print(f"\n原始数据统计:")
    print(f"  总细胞数: {adata.n_obs:,}")
    print(f"  总基因数: {adata.n_vars:,}")
    print(f"  数据集数: {adata.obs['dataset_id'].nunique()}")
    print(f"  样本数: {adata.obs['sample_id'].nunique()}")
    
    # ========================================================================
    # 第二部分：质控和预处理
    # ========================================================================
    print("\n" + "="*80)
    print("第二部分：质控和预处理")
    print("="*80)
    
    # 2.1 计算质控指标
    print("\n2.1 计算质控指标...")
    adata = calculate_qc_metrics(adata)
    
    # 2.2 绘制质控图
    print("\n2.2 绘制质控分布图...")
    plot_qc_metrics(adata, FIGURE_DIR)
    
    # 2.3 过滤低质量细胞和基因
    print("\n2.3 过滤低质量细胞和基因...")
    adata = filter_cells_and_genes(adata, QC_PARAMS)
    
    # 2.4 归一化和标准化
    print("\n2.4 归一化和标准化...")
    adata = normalize_and_scale(adata, n_top_genes=PROCESSING_PARAMS['n_top_genes'])
    
    # 2.5 批次效应校正
    print("\n2.5 批次效应校正...")
    adata = batch_correction(adata, batch_key='dataset_id')
    
    # 保存预处理后的数据
    print("\n保存预处理数据...")
    adata.write(os.path.join(DATA_DIR, 'adata_preprocessed.h5ad'))
    print(f"  已保存至: {DATA_DIR}/adata_preprocessed.h5ad")
    
    # ========================================================================
    # 第三部分：聚类和细胞类型注释
    # ========================================================================
    print("\n" + "="*80)
    print("第三部分：聚类和细胞类型注释")
    print("="*80)
    
    # 3.1 聚类
    print("\n3.1 聚类分析...")
    adata = clustering(adata, resolution=PROCESSING_PARAMS['resolution'], use_harmony=True)
    
    # 3.2 计算marker评分
    print("\n3.2 计算marker基因评分...")
    adata, marker_scores, marker_scores_zscore, marker_info = calculate_marker_scores(adata, marker_dict)
    
    # 3.3 注释细胞类型
    print("\n3.3 注释细胞类型...")
    adata, annotation_details, cluster_specificity, cluster_mean = annotate_clusters(
        adata, 
        marker_scores,
        marker_scores_zscore,
        marker_info,
        min_cluster_size=PROCESSING_PARAMS['min_cluster_size'],
        min_specificity=0.3,
        min_mean_zscore=0.5
    )
    
    # 保存注释详情
    annotation_details.to_csv(os.path.join(OUTPUT_DIR, 'annotation_details.csv'), index=False)
    cluster_specificity.to_csv(os.path.join(OUTPUT_DIR, 'cluster_specificity.csv'))
    cluster_mean.to_csv(os.path.join(OUTPUT_DIR, 'cluster_mean_scores.csv'))
    
    # 3.4 绘制注释结果
    print("\n3.4 绘制注释结果...")
    plot_annotation_results(adata, FIGURE_DIR, cluster_specificity, cluster_mean)
    
    # 保存注释后的数据
    print("\n保存注释数据...")
    adata.write(os.path.join(DATA_DIR, 'adata_annotated.h5ad'))
    print(f"  已保存至: {DATA_DIR}/adata_annotated.h5ad")
    
    # 输出细胞类型统计
    print("\n细胞类型统计:")
    celltype_stats = adata.obs.groupby(['dataset_id', 'celltype']).size().unstack(fill_value=0)
    print(celltype_stats)
    celltype_stats.to_csv(os.path.join(OUTPUT_DIR, 'celltype_statistics.csv'))
    
    # ========================================================================
    # 第四部分：假设检验分析
    # ========================================================================
    print("\n" + "="*80)
    print("第四部分：假设检验分析")
    print("="*80)
    
    # 4.1 H1: 组成偏倚分析
    print("\n4.1 H1: 组成偏倚分析...")
    h1_results = h1_composition_bias(adata, FIGURE_DIR)
    
    # 4.2 H2: 状态偏倚分析
    print("\n4.2 H2: 状态偏倚分析...")
    h2_results = h2_state_bias(adata, FIGURE_DIR, n_hvgs=2000)
    
    # 4.3 H3: 网络偏倚分析（改进版）
    print("\n4.3 H3: 网络偏倚分析（改进版）...")
    h3_results = h3_network_bias_improved(adata, FIGURE_DIR, n_hvgs=500)
    
    # ========================================================================
    # 第五部分：Normal-Normal差异分析
    # ========================================================================
    print("\n" + "="*80)
    print("第五部分：Normal-Normal差异分析")
    print("="*80)
    
    print("\n5.1 Normal-Normal差异表达分析...")
    de_results = normal_normal_differential_analysis(adata, FIGURE_DIR)
    
    # ========================================================================
    # 第六部分：生成整合结果表格
    # ========================================================================
    print("\n" + "="*80)
    print("第六部分：生成整合结果表格")
    print("="*80)
    
    result_tables = generate_all_result_tables(h1_results, h2_results, h3_results, de_results, OUTPUT_DIR)
    
    # ========================================================================
    # 第七部分：生成综合报告
    # ========================================================================
    print("\n" + "="*80)
    print("第七部分：生成综合报告")
    print("="*80)
    
    generate_summary_report(adata, h1_results, h2_results, h3_results, de_results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
    print(f"  - 图表: {FIGURE_DIR}/")
    print(f"  - 数据: {DATA_DIR}/")
    print(f"  - 结果表格: {OUTPUT_DIR}/Table_*.csv")
    print(f"  - 报告: {OUTPUT_DIR}/analysis_report.txt")


def generate_summary_report(adata, h1_results, h2_results, h3_results, de_results):
    """
    生成综合分析报告（简化版，详细数据见表格文件）
    """
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("单细胞RNA-seq数据分析报告：选择偏倚研究\n")
        f.write("Single-cell RNA-seq Analysis Report: Selection Bias Study\n")
        f.write("="*80 + "\n\n")
        f.write("注：详细数据请查看对应的表格文件 (Table_*.csv)\n\n")
        
        # 数据概览
        f.write("一、数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"总细胞数: {adata.n_obs:,}\n")
        f.write(f"总基因数: {adata.n_vars:,}\n")
        f.write(f"数据集数: {adata.obs['dataset_id'].nunique()}\n")
        f.write(f"样本数: {adata.obs['sample_id'].nunique()}\n")
        f.write(f"细胞类型数: {adata.obs['celltype'].nunique()}\n\n")
        
        f.write("各数据集信息:\n")
        for dataset_id, config in DATASETS.items():
            n_cells = (adata.obs['dataset_id'] == dataset_id).sum()
            f.write(f"  {dataset_id}: {config['disease']}\n")
            f.write(f"    细胞数: {n_cells:,}\n")
            f.write(f"    Normal样本数: {config['normal_count']}\n")
        f.write("\n")
        
        f.write("各细胞类型细胞数:\n")
        for celltype, count in adata.obs['celltype'].value_counts().items():
            f.write(f"  {celltype}: {count:,}\n")
        f.write("\n\n")
        
        # H1结果
        f.write("二、H1: 组成偏倚分析结果\n")
        f.write("-"*80 + "\n")
        f.write("详细数据见: Table_H1_Overall_Statistics.csv 和 Table_H1_CellType_Details.csv\n\n")
        
        if 'chi2_test' in h1_results:
            chi2_result = h1_results['chi2_test']
            f.write(f"卡方检验: χ² = {chi2_result['chi2']:.2f}, p = {chi2_result['pvalue']:.2e}\n")
            f.write(f"结论: {'显著差异 (p < 0.05)' if chi2_result['significant'] else '无显著差异'}\n\n")
        
        if 'js_divergence' in h1_results:
            js_df = h1_results['js_divergence']
            f.write(f"Jensen-Shannon散度范围: {js_df['js_divergence'].min():.3f} - {js_df['js_divergence'].max():.3f}\n\n")
        
        if 'celltype_variance' in h1_results:
            celltype_var = h1_results['celltype_variance']
            sig_celltypes = celltype_var[celltype_var['pvalue_adj'] < 0.05]
            f.write(f"显著差异的细胞类型: {len(sig_celltypes)}/{len(celltype_var)}\n\n")
        
        f.write("H1总结: ")
        if h1_results.get('chi2_test', {}).get('significant', False):
            f.write("细胞组成在数据集间存在显著差异，存在组成偏倚。\n\n\n")
        else:
            f.write("细胞组成在数据集间无显著差异。\n\n\n")
        
        # H2结果
        f.write("三、H2: 状态偏倚分析结果\n")
        f.write("-"*80 + "\n")
        f.write("详细数据见: Table_H2_State_Bias.csv\n\n")
        
        if len(h2_results) > 0:
            h2_df = pd.DataFrame(h2_results).T
            f.write(f"分析的细胞类型数: {len(h2_df)}\n\n")
            
            if 'variance_explained_by_dataset' in h2_df.columns:
                mean_var = h2_df['variance_explained_by_dataset'].mean()
                f.write(f"数据集平均解释方差比例 (R²): {mean_var:.2%}\n")
                f.write(f"范围: {h2_df['variance_explained_by_dataset'].min():.2%} - {h2_df['variance_explained_by_dataset'].max():.2%}\n\n")
                
                # 统计显著的细胞类型
                if 'permanova_p' in h2_df.columns:
                    sig_count = (h2_df['permanova_p'] < 0.05).sum()
                    f.write(f"PERMANOVA显著的细胞类型: {sig_count}/{len(h2_df)}\n\n")
                
                f.write("H2总结: ")
                if mean_var > 0.1:
                    f.write("数据集解释了大量方差 (R² > 10%)，存在显著的状态偏倚。\n\n\n")
                else:
                    f.write("数据集解释的方差较少 (R² < 10%)。\n\n\n")
        else:
            f.write("未能完成H2分析。\n\n\n")
        
        # H3结果
        f.write("四、H3: 网络偏倚分析结果 (WGCNA)\n")
        f.write("-"*80 + "\n")
        f.write("详细数据见: Table_H3_Network_Bias.csv\n\n")
        
        if len(h3_results) > 0:
            f.write(f"分析的细胞类型数: {len(h3_results)}\n\n")
            
            # 计算总体统计
            all_pres = [res['mean_preservation'] for res in h3_results.values()]
            mean_pres_all = np.mean(all_pres)
            
            f.write(f"总体平均模块保守性: {mean_pres_all:.3f}\n")
            f.write(f"范围: {min(all_pres):.3f} - {max(all_pres):.3f}\n\n")
            
            # 统计偏倚程度
            if 'bias_level' in list(h3_results.values())[0]:
                bias_levels = [res['bias_level'] for res in h3_results.values()]
                n_low = bias_levels.count('低')
                n_medium = bias_levels.count('中等')
                n_high = bias_levels.count('高')
                
                f.write(f"网络偏倚程度分布:\n")
                f.write(f"  低偏倚: {n_low} 个细胞类型\n")
                f.write(f"  中等偏倚: {n_medium} 个细胞类型\n")
                f.write(f"  高偏倚: {n_high} 个细胞类型\n\n")
            
            f.write("H3总结: ")
            if mean_pres_all > 0.7:
                f.write("WGCNA模块在数据集间高度保守，网络偏倚较弱。\n\n\n")
            elif mean_pres_all > 0.5:
                f.write("WGCNA模块在数据集间中等保守，存在一定网络偏倚。\n\n\n")
            else:
                f.write("WGCNA模块在数据集间保守性较低 (<0.5)，存在显著网络偏倚。\n\n\n")
        else:
            f.write("未能完成H3分析。\n\n\n")
        
        # Normal-Normal差异
        f.write("五、Normal-Normal差异表达分析结果\n")
        f.write("-"*80 + "\n")
        f.write("详细数据见: Table_DE_Overall_Statistics.csv 和 Table_DE_CellType_Details.csv\n\n")
        
        if len(de_results) > 0:
            # 收集所有结果
            all_de_data = []
            for celltype, res_dict in de_results.items():
                if isinstance(res_dict, dict) and 'de_summary' in res_dict:
                    all_de_data.append(res_dict['de_summary'])
            
            if len(all_de_data) > 0:
                all_de = pd.concat(all_de_data, ignore_index=True)
                
                f.write(f"总比较次数: {len(all_de)}\n")
                f.write(f"平均差异基因数: {all_de['n_sig_genes'].mean():.1f}\n")
                f.write(f"差异基因数范围: {all_de['n_sig_genes'].min():.0f} - {all_de['n_sig_genes'].max():.0f}\n\n")
                
                # 异质性评价
                mean_degs = all_de['n_sig_genes'].mean()
                f.write("异质性评价: ")
                if mean_degs > 200:
                    f.write("高异质性 (平均>200个DEGs)\n")
                    f.write("结论: Normal样本间存在大量差异表达基因，提示显著的样本间异质性。\n\n\n")
                elif mean_degs > 100:
                    f.write("中等异质性 (平均100-200个DEGs)\n")
                    f.write("结论: Normal样本间存在中等数量差异表达基因，提示一定程度的样本间异质性。\n\n\n")
                else:
                    f.write("低异质性 (平均<100个DEGs)\n")
                    f.write("结论: Normal样本间差异表达基因较少，样本间异质性较低。\n\n\n")
            else:
                f.write("未能完成差异分析。\n\n\n")
        else:
            f.write("未能完成差异分析。\n\n\n")
        
        # 总结
        f.write("六、总体结论\n")
        f.write("-"*80 + "\n")
        f.write("本研究通过单细胞RNA-seq数据分析，系统评估了不同研究间normal样本的异质性，\n")
        f.write("包括组成偏倚、状态偏倚和网络偏倚三个维度。\n\n")
        
        f.write("主要发现:\n")
        
        # H1
        if h1_results.get('chi2_test', {}).get('significant', False):
            f.write("1. 组成偏倚: 不同研究的normal样本在细胞类型组成上存在显著差异。\n")
        else:
            f.write("1. 组成偏倚: 不同研究的normal样本在细胞类型组成上较为一致。\n")
        
        # H2
        if len(h2_results) > 0:
            h2_df = pd.DataFrame(h2_results).T
            if 'variance_explained_by_dataset' in h2_df.columns:
                mean_var = h2_df['variance_explained_by_dataset'].mean()
                if mean_var > 0.1:
                    f.write("2. 状态偏倚: 同一细胞类型内，研究来源解释了大量转录组方差。\n")
                else:
                    f.write("2. 状态偏倚: 同一细胞类型内，研究来源解释的转录组方差较少。\n")
        
        # H3
        if len(h3_results) > 0:
            sample_result = list(h3_results.values())[0]
            
            if 'bias_level' in sample_result:
                # 新方法
                mean_pres_all = np.mean([res['mean_preservation'] for res in h3_results.values()])
                if mean_pres_all < 0.7:
                    f.write("3. 网络偏倚: 基因共表达网络在不同研究间保守性较低。\n")
                else:
                    f.write("3. 网络偏倚: 基因共表达网络在不同研究间保守性较高。\n")
            elif 'n_modules' in sample_result:
                # 旧方法
                mean_pres_all = np.mean([res['mean_preservation'] for res in h3_results.values()])
                if mean_pres_all < 0.7:
                    f.write("3. 网络偏倚: 基因共表达网络在不同研究间保守性较低。\n")
                else:
                    f.write("3. 网络偏倚: 基因共表达网络在不同研究间保守性较高。\n")
        
        f.write("\n")
        f.write("这些发现对于理解入院率偏倚和跨研究数据整合具有重要意义。\n")
        f.write("建议在进行跨研究比较时，充分考虑这些潜在的偏倚来源。\n\n")
        
        f.write("="*80 + "\n")
        f.write("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    
    print(f"\n综合报告已保存至: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
