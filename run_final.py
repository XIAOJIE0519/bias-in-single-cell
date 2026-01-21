"""
最终版本：H3网络偏倚分析（WGCNA） + Normal-Normal差异分析
从已注释的数据开始运行
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入配置和分析模块
from config import *
from utils_network_analysis_improved import (
    h3_network_bias_improved,
    normal_normal_differential_analysis
)

# 设置随机种子
np.random.seed(RANDOM_SEED)

# 设置scanpy参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sc.settings.figdir = FIGURE_DIR

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║           H3网络偏倚分析（WGCNA） + Normal-Normal差异分析                   ║
║           （从已注释的数据开始）                                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")


def main():
    """
    主分析流程：H3（WGCNA）和Normal-Normal差异分析
    """
    
    # ========================================================================
    # 1. 加载已注释的数据
    # ========================================================================
    print("\n" + "="*80)
    print("1. 加载已注释的数据")
    print("="*80)
    
    adata_path = os.path.join(DATA_DIR, 'adata_annotated.h5ad')
    
    if not os.path.exists(adata_path):
        print(f"\n错误: 未找到注释数据文件: {adata_path}")
        print("请先运行 main_analysis.py 完成前面的步骤")
        sys.exit(1)
    
    print(f"\n从文件加载: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    print(f"\n数据统计:")
    print(f"  总细胞数: {adata.n_obs:,}")
    print(f"  总基因数: {adata.n_vars:,}")
    print(f"  数据集数: {adata.obs['dataset_id'].nunique()}")
    print(f"  细胞类型数: {adata.obs['celltype'].nunique()}")
    
    print(f"\n各细胞类型细胞数:")
    for celltype, count in adata.obs['celltype'].value_counts().items():
        print(f"  {celltype}: {count:,}")
    
    # ========================================================================
    # 2. H3: 网络偏倚分析（WGCNA方法）
    # ========================================================================
    print("\n" + "="*80)
    print("2. H3: 网络偏倚分析（WGCNA方法）")
    print("="*80)
    
    print("\nWGCNA分析流程:")
    print("  1. 选择高变基因（每个细胞类型500个）")
    print("  2. 构建加权基因共表达网络（软阈值=6）")
    print("  3. 计算拓扑重叠矩阵（TOM）")
    print("  4. 层次聚类识别基因模块")
    print("  5. 比较不同数据集间的模块保守性")
    print("  6. 评估网络偏倚程度")
    
    try:
        h3_results = h3_network_bias_improved(
            adata, 
            FIGURE_DIR, 
            n_hvgs=500,      # 每个细胞类型使用500个高变基因
            soft_power=6     # WGCNA软阈值
        )
        
        print("\n✓ H3 WGCNA分析完成")
        
    except Exception as e:
        print(f"\n✗ H3分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        h3_results = {}
    
    # ========================================================================
    # 3. Normal-Normal差异表达分析
    # ========================================================================
    print("\n" + "="*80)
    print("3. Normal-Normal 差异表达分析")
    print("="*80)
    
    print("\n分析策略:")
    print("  1. 对每种细胞类型，两两比较不同数据集的Normal样本")
    print("  2. 使用Wilcoxon秩和检验识别差异表达基因")
    print("  3. 筛选标准: FDR < 0.05, |logFC| > 0.5")
    print("  4. 统计差异表达基因数量，评估样本间异质性")
    
    try:
        de_results = normal_normal_differential_analysis(adata, FIGURE_DIR)
        
        print("\n✓ Normal-Normal差异分析完成")
        
    except Exception as e:
        print(f"\n✗ Normal-Normal差异分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        de_results = {}
    
    # ========================================================================
    # 4. 生成综合报告
    # ========================================================================
    print("\n" + "="*80)
    print("4. 生成综合报告")
    print("="*80)
    
    generate_comprehensive_report(adata, h3_results, de_results)
    
    # ========================================================================
    # 5. 总结
    # ========================================================================
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    
    print(f"\n所有结果已保存至: {OUTPUT_DIR}/")
    print(f"\n生成的文件:")
    print(f"  📊 图表目录: {FIGURE_DIR}/")
    print(f"     • H3_wgcna_overall_summary.png - H3总体汇总图")
    print(f"     • H3_wgcna_summary.csv - H3汇总数据")
    print(f"     • H3_{{celltype}}_wgcna_analysis.png - 各细胞类型H3分析图")
    print(f"     • Normal_Normal_DE_overall_summary.png - 差异分析总体汇总图")
    print(f"     • Normal_Normal_DE_summary.csv - 差异分析汇总数据")
    print(f"     • DE_{{celltype}}_summary.png - 各细胞类型差异分析图")
    print(f"     • DE_{{celltype}}_{{dataset1}}_vs_{{dataset2}}.csv - 详细差异基因列表")
    print(f"\n  📄 综合报告: {OUTPUT_DIR}/Final_Analysis_Report.txt")
    print("\n" + "="*80)


def generate_comprehensive_report(adata, h3_results, de_results):
    """
    生成H3（WGCNA）和差异分析的综合报告
    """
    report_path = os.path.join(OUTPUT_DIR, 'Final_Analysis_Report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H3网络偏倚分析（WGCNA） + Normal-Normal差异分析 - 综合报告\n")
        f.write("H3 Network Bias (WGCNA) + Normal-Normal Differential Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        # ====================================================================
        # 一、数据概览
        # ====================================================================
        f.write("一、数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"总细胞数: {adata.n_obs:,}\n")
        f.write(f"总基因数: {adata.n_vars:,}\n")
        f.write(f"数据集数: {adata.obs['dataset_id'].nunique()}\n")
        f.write(f"细胞类型数: {adata.obs['celltype'].nunique()}\n\n")
        
        f.write("各数据集细胞数:\n")
        for dataset, count in adata.obs['dataset_id'].value_counts().items():
            f.write(f"  {dataset}: {count:,}\n")
        f.write("\n")
        
        f.write("各细胞类型细胞数:\n")
        for celltype, count in adata.obs['celltype'].value_counts().items():
            f.write(f"  {celltype}: {count:,}\n")
        f.write("\n\n")
        
        # ====================================================================
        # 二、H3: 网络偏倚分析结果（WGCNA方法）
        # ====================================================================
        f.write("二、H3: 网络偏倚分析结果（WGCNA方法）\n")
        f.write("-"*80 + "\n")
        f.write("方法: 加权基因共表达网络分析（WGCNA）\n")
        f.write("  - 每个细胞类型选择500个高变基因\n")
        f.write("  - 使用软阈值（power=6）构建加权网络\n")
        f.write("  - 计算拓扑重叠矩阵（TOM）\n")
        f.write("  - 层次聚类识别基因模块\n")
        f.write("  - 比较不同数据集间的模块保守性\n\n")
        
        if len(h3_results) > 0:
            f.write(f"成功分析的细胞类型数: {len(h3_results)}\n\n")
            
            # 总体统计
            all_pres = [res['mean_preservation'] for res in h3_results.values()]
            overall_pres = np.mean(all_pres)
            
            f.write(f"总体平均模块保守性: {overall_pres:.3f}\n\n")
            
            f.write("各细胞类型的WGCNA模块保守性:\n")
            f.write(f"{'细胞类型':<40} {'平均保守性':<15} {'偏倚程度':<15} {'比较次数':<10}\n")
            f.write("-"*80 + "\n")
            
            for celltype, res in sorted(h3_results.items(), 
                                       key=lambda x: x[1]['mean_preservation'], 
                                       reverse=True):
                f.write(f"{celltype:<40} {res['mean_preservation']:<15.3f} "
                       f"{res['bias_level']:<15} {res['n_comparisons']:<10}\n")
            
            f.write("\n")
            
            # 统计偏倚程度分布
            bias_levels = [res['bias_level'] for res in h3_results.values()]
            n_low = bias_levels.count('低')
            n_medium = bias_levels.count('中等')
            n_high = bias_levels.count('高')
            
            f.write("网络偏倚程度分布:\n")
            f.write(f"  低偏倚 (保守性>0.7): {n_low} 个细胞类型\n")
            f.write(f"  中等偏倚 (保守性0.5-0.7): {n_medium} 个细胞类型\n")
            f.write(f"  高偏倚 (保守性<0.5): {n_high} 个细胞类型\n\n")
            
            f.write("H3 WGCNA结论:\n")
            if overall_pres > 0.7:
                f.write("  ✓ WGCNA模块在数据集间高度保守\n")
                f.write("  → 网络偏倚较弱\n")
                f.write("  → 不同研究的基因共表达模式高度一致\n")
            elif overall_pres > 0.5:
                f.write("  ⚠ WGCNA模块在数据集间中等保守\n")
                f.write("  → 存在一定程度的网络偏倚\n")
                f.write("  → 不同研究的基因共表达模式存在差异\n")
            else:
                f.write("  ✗ WGCNA模块在数据集间保守性较低\n")
                f.write("  → 存在显著的网络偏倚\n")
                f.write("  → 不同研究的基因共表达模式差异显著\n")
            
            f.write(f"\n  • {n_high} 个细胞类型显示高网络偏倚\n")
            f.write(f"  • {n_medium} 个细胞类型显示中等网络偏倚\n")
            f.write(f"  • {n_low} 个细胞类型显示低网络偏倚\n")
        else:
            f.write("未能完成H3 WGCNA分析。\n")
        
        f.write("\n\n")
        
        # ====================================================================
        # 三、Normal-Normal差异表达分析结果
        # ====================================================================
        f.write("三、Normal-Normal差异表达分析结果\n")
        f.write("-"*80 + "\n")
        f.write("方法: Wilcoxon秩和检验\n")
        f.write("筛选标准: FDR < 0.05, |logFC| > 0.5\n\n")
        
        if len(de_results) > 0:
            # 汇总所有结果
            all_de_data = []
            for celltype, res_dict in de_results.items():
                if 'de_summary' in res_dict:
                    df = res_dict['de_summary']
                    all_de_data.append(df)
            
            if len(all_de_data) > 0:
                all_de_df = pd.concat(all_de_data, ignore_index=True)
                
                mean_degs = all_de_df['n_sig_genes'].mean()
                median_degs = all_de_df['n_sig_genes'].median()
                max_degs = all_de_df['n_sig_genes'].max()
                min_degs = all_de_df['n_sig_genes'].min()
                
                f.write(f"总比较次数: {len(all_de_df)}\n")
                f.write(f"平均差异基因数: {mean_degs:.1f}\n")
                f.write(f"中位数差异基因数: {median_degs:.1f}\n")
                f.write(f"最大差异基因数: {max_degs}\n")
                f.write(f"最小差异基因数: {min_degs}\n\n")
                
                # 各细胞类型统计
                celltype_stats = all_de_df.groupby('celltype')['n_sig_genes'].agg(['mean', 'std', 'count'])
                celltype_stats = celltype_stats.sort_values('mean', ascending=False)
                
                f.write("各细胞类型的平均差异基因数:\n")
                f.write(f"{'细胞类型':<40} {'平均DEGs':<15} {'标准差':<15} {'比较次数':<10}\n")
                f.write("-"*80 + "\n")
                
                for celltype, row in celltype_stats.iterrows():
                    f.write(f"{celltype:<40} {row['mean']:<15.1f} "
                           f"{row['std']:<15.1f} {int(row['count']):<10}\n")
                
                f.write("\n")
                
                f.write("Normal-Normal差异分析结论:\n")
                if mean_degs > 200:
                    f.write("  ✗ Normal样本间存在大量差异表达基因\n")
                    f.write("  → 提示显著的样本间异质性\n")
                    f.write("  → 不同研究的Normal样本在转录组水平差异显著\n")
                elif mean_degs > 100:
                    f.write("  ⚠ Normal样本间存在中等数量的差异表达基因\n")
                    f.write("  → 提示一定程度的样本间异质性\n")
                    f.write("  → 不同研究的Normal样本存在一定转录组差异\n")
                else:
                    f.write("  ✓ Normal样本间差异表达基因较少\n")
                    f.write("  → 样本间异质性较低\n")
                    f.write("  → 不同研究的Normal样本转录组较为一致\n")
                
                # 异质性最高的细胞类型
                top3 = celltype_stats.head(3)
                f.write("\n  异质性最高的细胞类型（Top 3）:\n")
                for idx, (celltype, row) in enumerate(top3.iterrows(), 1):
                    f.write(f"    {idx}. {celltype}: 平均 {row['mean']:.1f} 个DEGs\n")
        else:
            f.write("未能完成Normal-Normal差异分析。\n")
        
        f.write("\n\n")
        
        # ====================================================================
        # 四、综合结论与建议
        # ====================================================================
        f.write("四、综合结论与建议\n")
        f.write("-"*80 + "\n")
        f.write("本分析通过WGCNA网络偏倚和差异表达两个维度，全面评估了不同研究间\n")
        f.write("normal样本的异质性，为理解入院率偏倚提供了重要依据。\n\n")
        
        f.write("主要发现:\n\n")
        
        # H3结论
        if len(h3_results) > 0:
            all_pres = [res['mean_preservation'] for res in h3_results.values()]
            overall_pres = np.mean(all_pres)
            
            f.write("1. 网络层面（WGCNA）:\n")
            if overall_pres > 0.7:
                f.write("   ✓ 基因共表达网络在不同研究间高度保守\n")
                f.write("   ✓ 基因间的功能关联模式稳定\n")
                f.write("   → 网络层面的入院率偏倚较弱\n")
            elif overall_pres > 0.5:
                f.write("   ⚠ 基因共表达网络在不同研究间中等保守\n")
                f.write("   ⚠ 基因间的功能关联模式存在一定差异\n")
                f.write("   → 网络层面存在一定程度的入院率偏倚\n")
            else:
                f.write("   ✗ 基因共表达网络在不同研究间保守性较低\n")
                f.write("   ✗ 基因间的功能关联模式差异显著\n")
                f.write("   → 网络层面存在显著的入院率偏倚\n")
            f.write("\n")
        
        # 差异分析结论
        if len(de_results) > 0 and len(all_de_data) > 0:
            f.write("2. 表达层面（差异基因）:\n")
            if mean_degs > 200:
                f.write("   ✗ Normal样本间存在大量差异表达基因\n")
                f.write("   ✗ 转录组水平异质性显著\n")
                f.write("   → 表达层面的入院率偏倚显著\n")
            elif mean_degs > 100:
                f.write("   ⚠ Normal样本间存在中等数量差异表达基因\n")
                f.write("   ⚠ 转录组水平存在一定异质性\n")
                f.write("   → 表达层面存在一定程度的入院率偏倚\n")
            else:
                f.write("   ✓ Normal样本间差异表达基因较少\n")
                f.write("   ✓ 转录组水平异质性较低\n")
                f.write("   → 表达层面的入院率偏倚较弱\n")
            f.write("\n")
        
        f.write("研究意义:\n")
        f.write("• 揭示了不同研究间normal样本在网络和表达两个层面的异质性\n")
        f.write("• 为理解入院率偏倚的分子机制提供了重要线索\n")
        f.write("• 为跨研究数据整合和meta分析提供了参考依据\n\n")
        
        f.write("建议:\n")
        f.write("1. 在进行跨研究比较时，应充分考虑这些潜在的偏倚来源\n")
        f.write("2. 对于网络偏倚显著的细胞类型，建议进行批次效应校正\n")
        f.write("3. 对于差异基因较多的细胞类型，建议深入分析其生物学意义\n")
        f.write("4. 在meta分析中，建议使用稳健的统计方法处理异质性\n")
        f.write("5. 未来研究应关注导致这些差异的潜在因素（如样本采集、处理流程等）\n\n")
        
        f.write("="*80 + "\n")
        f.write("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ 综合报告已保存至: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
