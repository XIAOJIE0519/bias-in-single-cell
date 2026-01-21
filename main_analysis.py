"""
Main analysis script: Complete single-cell data analysis pipeline
Following Nature-level standards for rigorous analysis
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

# Import custom modules
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

# Set random seed
np.random.seed(RANDOM_SEED)

# Set scanpy parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=False)
sc.settings.figdir = FIGURE_DIR

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║           Single-cell RNA-seq Analysis: Selection Bias Study             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

def main():
    """
    Main analysis pipeline
    """
    
    # ========================================================================
    # Part 1: Data Loading
    # ========================================================================
    print("\n" + "="*80)
    print("Part 1: Data Loading")
    print("="*80)
    
    # 1.1 Load all datasets
    print("\n1.1 Loading all datasets...")
    adata = load_all_datasets(DATASETS, BASE_DIR)
    
    # 1.2 Load marker genes
    print("\n1.2 Loading marker gene annotations...")
    marker_dict = load_marker_genes(MARKER_FILE)
    
    # Save original data information
    print(f"\nOriginal data statistics:")
    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Total genes: {adata.n_vars:,}")
    print(f"  Number of datasets: {adata.obs['dataset_id'].nunique()}")
    print(f"  Number of samples: {adata.obs['sample_id'].nunique()}")
    
    # ========================================================================
    # Part 2: Quality Control and Preprocessing
    # ========================================================================
    print("\n" + "="*80)
    print("Part 2: Quality Control and Preprocessing")
    print("="*80)
    
    # 2.1 Calculate QC metrics
    print("\n2.1 Calculating QC metrics...")
    adata = calculate_qc_metrics(adata)
    
    # 2.2 Plot QC distributions
    print("\n2.2 Plotting QC distributions...")
    plot_qc_metrics(adata, FIGURE_DIR)
    
    # 2.3 Filter low-quality cells and genes
    print("\n2.3 Filtering low-quality cells and genes...")
    adata = filter_cells_and_genes(adata, QC_PARAMS)
    
    # 2.4 Normalization and scaling
    print("\n2.4 Normalization and scaling...")
    adata = normalize_and_scale(adata, n_top_genes=PROCESSING_PARAMS['n_top_genes'])
    
    # 2.5 Batch effect correction
    print("\n2.5 Batch effect correction...")
    adata = batch_correction(adata, batch_key='dataset_id')
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    adata.write(os.path.join(DATA_DIR, 'adata_preprocessed.h5ad'))
    print(f"  Saved to: {DATA_DIR}/adata_preprocessed.h5ad")
    
    # ========================================================================
    # Part 3: Clustering and Cell Type Annotation
    # ========================================================================
    print("\n" + "="*80)
    print("Part 3: Clustering and Cell Type Annotation")
    print("="*80)
    
    # 3.1 Clustering
    print("\n3.1 Clustering analysis...")
    adata = clustering(adata, resolution=PROCESSING_PARAMS['resolution'], use_harmony=True)
    
    # 3.2 Calculate marker scores
    print("\n3.2 Calculating marker gene scores...")
    adata, marker_scores, marker_scores_zscore, marker_info = calculate_marker_scores(adata, marker_dict)
    
    # 3.3 Annotate cell types
    print("\n3.3 Annotating cell types...")
    adata, annotation_details, cluster_specificity, cluster_mean = annotate_clusters(
        adata, 
        marker_scores,
        marker_scores_zscore,
        marker_info,
        min_cluster_size=PROCESSING_PARAMS['min_cluster_size'],
        min_specificity=0.3,
        min_mean_zscore=0.5
    )
    
    # Save annotation details
    annotation_details.to_csv(os.path.join(OUTPUT_DIR, 'annotation_details.csv'), index=False)
    cluster_specificity.to_csv(os.path.join(OUTPUT_DIR, 'cluster_specificity.csv'))
    cluster_mean.to_csv(os.path.join(OUTPUT_DIR, 'cluster_mean_scores.csv'))
    
    # 3.4 Plot annotation results
    print("\n3.4 Plotting annotation results...")
    plot_annotation_results(adata, FIGURE_DIR, cluster_specificity, cluster_mean)
    
    # Save annotated data
    print("\nSaving annotated data...")
    adata.write(os.path.join(DATA_DIR, 'adata_annotated.h5ad'))
    print(f"  Saved to: {DATA_DIR}/adata_annotated.h5ad")
    
    # Output cell type statistics
    print("\nCell type statistics:")
    celltype_stats = adata.obs.groupby(['dataset_id', 'celltype']).size().unstack(fill_value=0)
    print(celltype_stats)
    celltype_stats.to_csv(os.path.join(OUTPUT_DIR, 'celltype_statistics.csv'))
    
    # ========================================================================
    # Part 4: Hypothesis Testing Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("Part 4: Hypothesis Testing Analysis")
    print("="*80)
    
    # 4.1 H1: Composition bias analysis
    print("\n4.1 H1: Composition bias analysis...")
    h1_results = h1_composition_bias(adata, FIGURE_DIR)
    
    # 4.2 H2: State bias analysis
    print("\n4.2 H2: State bias analysis...")
    h2_results = h2_state_bias(adata, FIGURE_DIR, n_hvgs=2000)
    
    # 4.3 H3: Network bias analysis (improved version)
    print("\n4.3 H3: Network bias analysis (improved version)...")
    h3_results = h3_network_bias_improved(adata, FIGURE_DIR, n_hvgs=500)
    
    # ========================================================================
    # Part 5: Normal-Normal Differential Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("Part 5: Normal-Normal Differential Analysis")
    print("="*80)
    
    print("\n5.1 Normal-Normal differential expression analysis...")
    de_results = normal_normal_differential_analysis(adata, FIGURE_DIR)
    
    # ========================================================================
    # Part 6: Generate Integrated Result Tables
    # ========================================================================
    print("\n" + "="*80)
    print("Part 6: Generate Integrated Result Tables")
    print("="*80)
    
    result_tables = generate_all_result_tables(h1_results, h2_results, h3_results, de_results, OUTPUT_DIR)
    
    # ========================================================================
    # Part 7: Generate Summary Report
    # ========================================================================
    print("\n" + "="*80)
    print("Part 7: Generate Summary Report")
    print("="*80)
    
    generate_summary_report(adata, h1_results, h2_results, h3_results, de_results)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print(f"  - Figures: {FIGURE_DIR}/")
    print(f"  - Data: {DATA_DIR}/")
    print(f"  - Result tables: {OUTPUT_DIR}/Table_*.csv")
    print(f"  - Report: {OUTPUT_DIR}/analysis_report.txt")


def generate_summary_report(adata, h1_results, h2_results, h3_results, de_results):
    """
    Generate comprehensive analysis report (simplified version, detailed data in table files)
    """
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Single-cell RNA-seq Analysis Report: Selection Bias Study\n")
        f.write("="*80 + "\n\n")
        f.write("Note: Detailed data available in corresponding table files (Table_*.csv)\n\n")
        
        # Data overview
        f.write("I. Data Overview\n")
        f.write("-"*80 + "\n")
        f.write(f"Total cells: {adata.n_obs:,}\n")
        f.write(f"Total genes: {adata.n_vars:,}\n")
        f.write(f"Number of datasets: {adata.obs['dataset_id'].nunique()}\n")
        f.write(f"Number of samples: {adata.obs['sample_id'].nunique()}\n")
        f.write(f"Number of cell types: {adata.obs['celltype'].nunique()}\n\n")
        
        f.write("Dataset information:\n")
        for dataset_id, config in DATASETS.items():
            n_cells = (adata.obs['dataset_id'] == dataset_id).sum()
            f.write(f"  {dataset_id}: {config['disease']}\n")
            f.write(f"    Cells: {n_cells:,}\n")
            f.write(f"    Normal samples: {config['normal_count']}\n")
        f.write("\n")
        
        f.write("Cell counts by cell type:\n")
        for celltype, count in adata.obs['celltype'].value_counts().items():
            f.write(f"  {celltype}: {count:,}\n")
        f.write("\n\n")
        
        # H1 results
        f.write("II. H1: Composition Bias Analysis Results\n")
        f.write("-"*80 + "\n")
        f.write("Detailed data: Table_H1_Overall_Statistics.csv and Table_H1_CellType_Details.csv\n\n")
        
        if 'chi2_test' in h1_results:
            chi2_result = h1_results['chi2_test']
            f.write(f"Chi-square test: χ² = {chi2_result['chi2']:.2f}, p = {chi2_result['pvalue']:.2e}\n")
            f.write(f"Conclusion: {'Significant difference (p < 0.05)' if chi2_result['significant'] else 'No significant difference'}\n\n")
        
        if 'js_divergence' in h1_results:
            js_df = h1_results['js_divergence']
            f.write(f"Jensen-Shannon divergence range: {js_df['js_divergence'].min():.3f} - {js_df['js_divergence'].max():.3f}\n\n")
        
        if 'celltype_variance' in h1_results:
            celltype_var = h1_results['celltype_variance']
            sig_celltypes = celltype_var[celltype_var['pvalue_adj'] < 0.05]
            f.write(f"Cell types with significant differences: {len(sig_celltypes)}/{len(celltype_var)}\n\n")
        
        f.write("H1 Summary: ")
        if h1_results.get('chi2_test', {}).get('significant', False):
            f.write("Significant composition differences exist across datasets, indicating composition bias.\n\n\n")
        else:
            f.write("No significant composition differences across datasets.\n\n\n")
        
        # H2 results
        f.write("III. H2: State Bias Analysis Results\n")
        f.write("-"*80 + "\n")
        f.write("Detailed data: Table_H2_State_Bias.csv\n\n")
        
        if len(h2_results) > 0:
            h2_df = pd.DataFrame(h2_results).T
            f.write(f"Number of cell types analyzed: {len(h2_df)}\n\n")
            
            if 'variance_explained_by_dataset' in h2_df.columns:
                mean_var = h2_df['variance_explained_by_dataset'].mean()
                f.write(f"Mean variance explained by dataset (R²): {mean_var:.2%}\n")
                f.write(f"Range: {h2_df['variance_explained_by_dataset'].min():.2%} - {h2_df['variance_explained_by_dataset'].max():.2%}\n\n")
                
                if 'permanova_p' in h2_df.columns:
                    sig_count = (h2_df['permanova_p'] < 0.05).sum()
                    f.write(f"Cell types with significant PERMANOVA: {sig_count}/{len(h2_df)}\n\n")
                
                f.write("H2 Summary: ")
                if mean_var > 0.1:
                    f.write("Dataset explains substantial variance (R² > 10%), indicating significant state bias.\n\n\n")
                else:
                    f.write("Dataset explains minimal variance (R² < 10%).\n\n\n")
        else:
            f.write("H2 analysis not completed.\n\n\n")
        
        # H3 results
        f.write("IV. H3: Network Bias Analysis Results (WGCNA)\n")
        f.write("-"*80 + "\n")
        f.write("Detailed data: Table_H3_Network_Bias.csv\n\n")
        
        if len(h3_results) > 0:
            f.write(f"Number of cell types analyzed: {len(h3_results)}\n\n")
            
            all_pres = [res['mean_preservation'] for res in h3_results.values()]
            mean_pres_all = np.mean(all_pres)
            
            f.write(f"Overall mean module preservation: {mean_pres_all:.3f}\n")
            f.write(f"Range: {min(all_pres):.3f} - {max(all_pres):.3f}\n\n")
            
            if 'bias_level' in list(h3_results.values())[0]:
                bias_levels = [res['bias_level'] for res in h3_results.values()]
                n_low = bias_levels.count('Low')
                n_medium = bias_levels.count('Medium')
                n_high = bias_levels.count('High')
                
                f.write(f"Network bias level distribution:\n")
                f.write(f"  Low bias: {n_low} cell types\n")
                f.write(f"  Medium bias: {n_medium} cell types\n")
                f.write(f"  High bias: {n_high} cell types\n\n")
            
            f.write("H3 Summary: ")
            if mean_pres_all > 0.7:
                f.write("WGCNA modules highly preserved across datasets, weak network bias.\n\n\n")
            elif mean_pres_all > 0.5:
                f.write("WGCNA modules moderately preserved across datasets, moderate network bias.\n\n\n")
            else:
                f.write("WGCNA modules poorly preserved across datasets (<0.5), significant network bias.\n\n\n")
        else:
            f.write("H3 analysis not completed.\n\n\n")
        
        # DE results
        f.write("V. Normal-Normal Differential Expression Analysis Results\n")
        f.write("-"*80 + "\n")
        f.write("Detailed data: Table_DE_Overall_Statistics.csv and Table_DE_CellType_Details.csv\n\n")
        
        if len(de_results) > 0:
            all_de_data = []
            for celltype, res_dict in de_results.items():
                if isinstance(res_dict, dict) and 'de_summary' in res_dict:
                    all_de_data.append(res_dict['de_summary'])
            
            if len(all_de_data) > 0:
                all_de = pd.concat(all_de_data, ignore_index=True)
                
                f.write(f"Total comparisons: {len(all_de)}\n")
                f.write(f"Mean DEGs: {all_de['n_sig_genes'].mean():.1f}\n")
                f.write(f"DEG range: {all_de['n_sig_genes'].min():.0f} - {all_de['n_sig_genes'].max():.0f}\n\n")
                
                mean_degs = all_de['n_sig_genes'].mean()
                f.write("Heterogeneity assessment: ")
                if mean_degs > 200:
                    f.write("High heterogeneity (mean >200 DEGs)\n")
                    f.write("Conclusion: Substantial DEGs between normal samples, indicating significant inter-sample heterogeneity.\n\n\n")
                elif mean_degs > 100:
                    f.write("Moderate heterogeneity (mean 100-200 DEGs)\n")
                    f.write("Conclusion: Moderate DEGs between normal samples, indicating some inter-sample heterogeneity.\n\n\n")
                else:
                    f.write("Low heterogeneity (mean <100 DEGs)\n")
                    f.write("Conclusion: Few DEGs between normal samples, low inter-sample heterogeneity.\n\n\n")
            else:
                f.write("Differential analysis not completed.\n\n\n")
        else:
            f.write("Differential analysis not completed.\n\n\n")
        
        # Summary
        f.write("VI. Overall Conclusions\n")
        f.write("-"*80 + "\n")
        f.write("This study systematically evaluated heterogeneity of normal samples across\n")
        f.write("different studies through single-cell RNA-seq analysis, including composition\n")
        f.write("bias, state bias, and network bias.\n\n")
        
        f.write("Key findings:\n")
        
        if h1_results.get('chi2_test', {}).get('significant', False):
            f.write("1. Composition bias: Significant differences in cell type composition across studies.\n")
        else:
            f.write("1. Composition bias: Consistent cell type composition across studies.\n")
        
        if len(h2_results) > 0:
            h2_df = pd.DataFrame(h2_results).T
            if 'variance_explained_by_dataset' in h2_df.columns:
                mean_var = h2_df['variance_explained_by_dataset'].mean()
                if mean_var > 0.1:
                    f.write("2. State bias: Study origin explains substantial transcriptomic variance within cell types.\n")
                else:
                    f.write("2. State bias: Study origin explains minimal transcriptomic variance within cell types.\n")
        
        if len(h3_results) > 0:
            mean_pres_all = np.mean([res['mean_preservation'] for res in h3_results.values()])
            if mean_pres_all < 0.7:
                f.write("3. Network bias: Gene co-expression networks poorly preserved across studies.\n")
            else:
                f.write("3. Network bias: Gene co-expression networks well preserved across studies.\n")
        
        f.write("\n")
        f.write("These findings have important implications for understanding selection bias\n")
        f.write("and cross-study data integration. Careful consideration of these potential\n")
        f.write("sources of bias is recommended when conducting cross-study comparisons.\n\n")
        
        f.write("="*80 + "\n")
        f.write("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    
    print(f"\nSummary report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
