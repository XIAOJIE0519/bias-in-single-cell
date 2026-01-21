"""
结果表格整理模块
将H1、H2、H3和差异分析的结果整理成规范的表格
"""
import pandas as pd
import numpy as np
import os


def generate_h1_tables(h1_results, output_dir):
    """
    生成H1组成偏倚的整合表格
    
    Parameters:
    -----------
    h1_results : dict
        H1分析结果
    output_dir : str
        输出目录
    
    Returns:
    --------
    table1 : DataFrame
        总体统计表
    table2 : DataFrame
        细胞类型详细表
    """
    print("\n生成H1结果表格...")
    
    # 表1: H1总体统计
    table1_data = []
    
    # 卡方检验
    if 'chi2_test' in h1_results:
        chi2_result = h1_results['chi2_test']
        table1_data.append({
            '检验方法': '卡方检验 (Chi-square test)',
            '统计量': f"{chi2_result['chi2']:.2f}",
            'P值': f"{chi2_result['pvalue']:.2e}",
            '自由度': f"{chi2_result['dof']}",
            '是否显著': '是' if chi2_result['significant'] else '否',
            '结论': '细胞组成在数据集间存在显著差异' if chi2_result['significant'] else '细胞组成在数据集间无显著差异'
        })
    
    # JS散度
    if 'js_divergence' in h1_results:
        js_df = h1_results['js_divergence']
        table1_data.append({
            '检验方法': 'Jensen-Shannon散度',
            '统计量': f"{js_df['js_divergence'].mean():.4f} (平均)",
            'P值': '-',
            '自由度': '-',
            '是否显著': '-',
            '结论': f"数据集间组成差异范围: {js_df['js_divergence'].min():.3f} - {js_df['js_divergence'].max():.3f}"
        })
    
    table1 = pd.DataFrame(table1_data)
    table1.to_csv(os.path.join(output_dir, 'Table_H1_Overall_Statistics.csv'), index=False, encoding='utf-8-sig')
    
    # 表2: 各细胞类型的组成差异
    if 'celltype_variance' in h1_results:
        table2 = h1_results['celltype_variance'].copy()
        table2 = table2.rename(columns={
            'celltype': '细胞类型',
            'h_statistic': 'Kruskal-Wallis H统计量',
            'pvalue': 'P值',
            'pvalue_adj': '校正后P值 (FDR)',
            'mean_pct': '平均比例 (%)',
            'std_pct': '标准差 (%)'
        })
        
        # 添加显著性标记
        table2['显著性'] = table2['校正后P值 (FDR)'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
        )
        
        # 排序
        table2 = table2.sort_values('校正后P值 (FDR)')
        
        # 格式化数值
        table2['Kruskal-Wallis H统计量'] = table2['Kruskal-Wallis H统计量'].apply(lambda x: f"{x:.2f}")
        table2['P值'] = table2['P值'].apply(lambda x: f"{x:.2e}")
        table2['校正后P值 (FDR)'] = table2['校正后P值 (FDR)'].apply(lambda x: f"{x:.2e}")
        table2['平均比例 (%)'] = table2['平均比例 (%)'].apply(lambda x: f"{x:.2f}")
        table2['标准差 (%)'] = table2['标准差 (%)'].apply(lambda x: f"{x:.2f}")
        
        table2.to_csv(os.path.join(output_dir, 'Table_H1_CellType_Details.csv'), index=False, encoding='utf-8-sig')
    else:
        table2 = None
    
    print(f"  ✓ H1表格已保存")
    print(f"    - Table_H1_Overall_Statistics.csv")
    print(f"    - Table_H1_CellType_Details.csv")
    
    return table1, table2


def generate_h2_tables(h2_results, output_dir):
    """
    生成H2状态偏倚的整合表格
    
    Parameters:
    -----------
    h2_results : dict
        H2分析结果
    output_dir : str
        输出目录
    
    Returns:
    --------
    table : DataFrame
        H2结果表
    """
    print("\n生成H2结果表格...")
    
    if len(h2_results) == 0:
        print("  ⚠ H2结果为空，跳过")
        return None
    
    # 转换为DataFrame
    table = pd.DataFrame(h2_results).T
    table.index.name = '细胞类型'
    table = table.reset_index()
    
    # 重命名列
    rename_dict = {
        '细胞类型': '细胞类型',
        'n_cells': '细胞数',
        'n_datasets': '数据集数',
        'variance_explained_by_dataset': '数据集解释方差比例 (R²)',
    }
    
    if 'permanova_f' in table.columns:
        rename_dict['permanova_f'] = 'PERMANOVA F统计量'
        rename_dict['permanova_p'] = 'PERMANOVA P值'
    
    if 'silhouette_score' in table.columns:
        rename_dict['silhouette_score'] = 'Silhouette系数'
    
    table = table.rename(columns=rename_dict)
    
    # 添加显著性和偏倚程度判断
    if 'PERMANOVA P值' in table.columns:
        table['PERMANOVA显著性'] = table['PERMANOVA P值'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
        )
    
    if '数据集解释方差比例 (R²)' in table.columns:
        table['状态偏倚程度'] = table['数据集解释方差比例 (R²)'].apply(
            lambda x: '高 (R²>15%)' if x > 0.15 else '中等 (R²=5-15%)' if x > 0.05 else '低 (R²<5%)'
        )
        
        # 排序
        table = table.sort_values('数据集解释方差比例 (R²)', ascending=False)
        
        # 格式化
        table['数据集解释方差比例 (R²)'] = table['数据集解释方差比例 (R²)'].apply(lambda x: f"{x:.2%}")
    
    if 'PERMANOVA F统计量' in table.columns:
        table['PERMANOVA F统计量'] = table['PERMANOVA F统计量'].apply(lambda x: f"{x:.4f}")
        table['PERMANOVA P值'] = table['PERMANOVA P值'].apply(lambda x: f"{x:.4f}")
    
    table.to_csv(os.path.join(output_dir, 'Table_H2_State_Bias.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  ✓ H2表格已保存")
    print(f"    - Table_H2_State_Bias.csv")
    
    return table


def generate_h3_tables(h3_results, output_dir):
    """
    生成H3网络偏倚的整合表格
    
    Parameters:
    -----------
    h3_results : dict
        H3分析结果
    output_dir : str
        输出目录
    
    Returns:
    --------
    table : DataFrame
        H3结果表
    """
    print("\n生成H3结果表格...")
    
    if len(h3_results) == 0:
        print("  ⚠ H3结果为空，跳过")
        return None
    
    # 转换为DataFrame
    table = pd.DataFrame(h3_results).T
    table.index.name = '细胞类型'
    table = table.reset_index()
    
    # 重命名列
    table = table.rename(columns={
        '细胞类型': '细胞类型',
        'mean_preservation': '平均模块保守性',
        'n_comparisons': '比较次数',
        'bias_level': '网络偏倚程度'
    })
    
    # 排序（按保守性从低到高，即偏倚从高到低）
    table = table.sort_values('平均模块保守性')
    
    # 添加解释
    table['保守性评价'] = table['平均模块保守性'].apply(
        lambda x: '高保守 (>0.7)' if x > 0.7 else '中等保守 (0.5-0.7)' if x > 0.5 else '低保守 (<0.5)'
    )
    
    # 格式化
    table['平均模块保守性'] = table['平均模块保守性'].apply(lambda x: f"{x:.3f}")
    
    table.to_csv(os.path.join(output_dir, 'Table_H3_Network_Bias.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  ✓ H3表格已保存")
    print(f"    - Table_H3_Network_Bias.csv")
    
    return table


def generate_de_tables(de_results, output_dir):
    """
    生成差异表达分析的整合表格
    
    Parameters:
    -----------
    de_results : dict
        差异分析结果
    output_dir : str
        输出目录
    
    Returns:
    --------
    table1 : DataFrame
        总体统计表
    table2 : DataFrame
        各细胞类型详细表
    """
    print("\n生成差异表达分析结果表格...")
    
    if len(de_results) == 0:
        print("  ⚠ 差异分析结果为空，跳过")
        return None, None
    
    # 收集所有比较结果
    all_comparisons = []
    
    for celltype, res_dict in de_results.items():
        if 'de_summary' in res_dict:
            all_comparisons.append(res_dict['de_summary'])
    
    if len(all_comparisons) == 0:
        print("  ⚠ 无有效的差异分析结果")
        return None, None
    
    all_de = pd.concat(all_comparisons, ignore_index=True)
    
    # 表1: 总体统计
    table1_data = []
    
    table1_data.append({
        '统计指标': '总比较次数',
        '数值': f"{len(all_de)}",
        '说明': '所有细胞类型的数据集两两比较总数'
    })
    
    table1_data.append({
        '统计指标': '平均差异基因数',
        '数值': f"{all_de['n_sig_genes'].mean():.1f}",
        '说明': '每次比较平均发现的显著差异基因数 (FDR<0.05, |log2FC|>0.5)'
    })
    
    table1_data.append({
        '统计指标': '中位数差异基因数',
        '数值': f"{all_de['n_sig_genes'].median():.1f}",
        '说明': '差异基因数的中位数'
    })
    
    table1_data.append({
        '统计指标': '差异基因数范围',
        '数值': f"{all_de['n_sig_genes'].min():.0f} - {all_de['n_sig_genes'].max():.0f}",
        '说明': '最少到最多的差异基因数'
    })
    
    # 异质性评价
    mean_degs = all_de['n_sig_genes'].mean()
    if mean_degs > 200:
        heterogeneity = '高异质性'
        interpretation = 'Normal样本间存在大量差异表达基因，提示显著的样本间异质性'
    elif mean_degs > 100:
        heterogeneity = '中等异质性'
        interpretation = 'Normal样本间存在中等数量差异表达基因，提示一定程度的样本间异质性'
    else:
        heterogeneity = '低异质性'
        interpretation = 'Normal样本间差异表达基因较少，样本间异质性较低'
    
    table1_data.append({
        '统计指标': '异质性评价',
        '数值': heterogeneity,
        '说明': interpretation
    })
    
    table1 = pd.DataFrame(table1_data)
    table1.to_csv(os.path.join(output_dir, 'Table_DE_Overall_Statistics.csv'), index=False, encoding='utf-8-sig')
    
    # 表2: 各细胞类型的详细统计
    celltype_stats = all_de.groupby('celltype').agg({
        'n_sig_genes': ['mean', 'std', 'min', 'max', 'count'],
        'n_up': 'mean',
        'n_down': 'mean'
    }).reset_index()
    
    celltype_stats.columns = [
        '细胞类型',
        '平均差异基因数',
        '标准差',
        '最小值',
        '最大值',
        '比较次数',
        '平均上调基因数',
        '平均下调基因数'
    ]
    
    # 排序
    celltype_stats = celltype_stats.sort_values('平均差异基因数', ascending=False)
    
    # 添加异质性评价
    celltype_stats['异质性程度'] = celltype_stats['平均差异基因数'].apply(
        lambda x: '高 (>200)' if x > 200 else '中等 (100-200)' if x > 100 else '低 (<100)'
    )
    
    # 格式化
    celltype_stats['平均差异基因数'] = celltype_stats['平均差异基因数'].apply(lambda x: f"{x:.1f}")
    celltype_stats['标准差'] = celltype_stats['标准差'].apply(lambda x: f"{x:.1f}")
    celltype_stats['最小值'] = celltype_stats['最小值'].apply(lambda x: f"{x:.0f}")
    celltype_stats['最大值'] = celltype_stats['最大值'].apply(lambda x: f"{x:.0f}")
    celltype_stats['比较次数'] = celltype_stats['比较次数'].apply(lambda x: f"{int(x)}")
    celltype_stats['平均上调基因数'] = celltype_stats['平均上调基因数'].apply(lambda x: f"{x:.1f}")
    celltype_stats['平均下调基因数'] = celltype_stats['平均下调基因数'].apply(lambda x: f"{x:.1f}")
    
    table2 = celltype_stats
    table2.to_csv(os.path.join(output_dir, 'Table_DE_CellType_Details.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  ✓ 差异表达分析表格已保存")
    print(f"    - Table_DE_Overall_Statistics.csv")
    print(f"    - Table_DE_CellType_Details.csv")
    
    return table1, table2


def generate_all_result_tables(h1_results, h2_results, h3_results, de_results, output_dir):
    """
    生成所有结果的整合表格
    
    Parameters:
    -----------
    h1_results : dict
        H1分析结果
    h2_results : dict
        H2分析结果
    h3_results : dict
        H3分析结果
    de_results : dict
        差异分析结果
    output_dir : str
        输出目录
    """
    print("\n" + "="*60)
    print("生成整合结果表格")
    print("="*60)
    
    # 生成各部分表格
    h1_table1, h1_table2 = generate_h1_tables(h1_results, output_dir)
    h2_table = generate_h2_tables(h2_results, output_dir)
    h3_table = generate_h3_tables(h3_results, output_dir)
    de_table1, de_table2 = generate_de_tables(de_results, output_dir)
    
    print("\n" + "="*60)
    print("所有结果表格生成完成！")
    print("="*60)
    print("\n生成的表格文件:")
    print("  H1 组成偏倚:")
    print("    - Table_H1_Overall_Statistics.csv (总体统计)")
    print("    - Table_H1_CellType_Details.csv (细胞类型详细)")
    print("\n  H2 状态偏倚:")
    print("    - Table_H2_State_Bias.csv (各细胞类型状态偏倚)")
    print("\n  H3 网络偏倚:")
    print("    - Table_H3_Network_Bias.csv (各细胞类型网络偏倚)")
    print("\n  差异表达分析:")
    print("    - Table_DE_Overall_Statistics.csv (总体统计)")
    print("    - Table_DE_CellType_Details.csv (细胞类型详细)")
    print("\n这些表格可以直接用于论文的补充材料！")
    print("="*60)
    
    return {
        'h1': (h1_table1, h1_table2),
        'h2': h2_table,
        'h3': h3_table,
        'de': (de_table1, de_table2)
    }
