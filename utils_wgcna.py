"""
WGCNA核心功能实现
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')


def calculate_correlation_matrix(expr_matrix, method='pearson'):
    """
    计算基因相关性矩阵，处理NaN值
    
    Parameters:
    -----------
    expr_matrix : ndarray
        表达矩阵 (cells x genes)
    method : str
        'pearson' 或 'spearman'
    
    Returns:
    --------
    corr_matrix : ndarray
        相关性矩阵
    """
    n_genes = expr_matrix.shape[1]
    corr_matrix = np.zeros((n_genes, n_genes))
    
    for i in range(n_genes):
        for j in range(i, n_genes):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                x = expr_matrix[:, i]
                y = expr_matrix[:, j]
                
                # 移除NaN值
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:  # 至少需要10个有效值
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
                    continue
                
                x_clean = x[mask]
                y_clean = y[mask]
                
                # 检查方差
                if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
                    continue
                
                try:
                    if method == 'spearman':
                        corr, _ = spearmanr(x_clean, y_clean)
                    else:
                        corr, _ = pearsonr(x_clean, y_clean)
                    
                    if np.isnan(corr):
                        corr = 0.0
                    
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                except:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
    
    return corr_matrix


def calculate_soft_threshold(corr_matrix, powers=range(1, 21)):
    """
    计算WGCNA的软阈值
    """
    scale_free_r2 = []
    
    for power in powers:
        adj_matrix = np.abs(corr_matrix) ** power
        k = adj_matrix.sum(axis=1)
        
        # 计算scale-free拓扑
        k_unique = np.unique(k)
        if len(k_unique) > 5:
            hist, bin_edges = np.histogram(k, bins=20)
            hist = hist[hist > 0]
            bin_centers = (bin_edges[:-1] + bin_edges[1:])[hist > 0] / 2
            
            if len(hist) > 2:
                log_k = np.log10(bin_centers + 1)
                log_p = np.log10(hist + 1)
                
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(log_k, log_p)
                    scale_free_r2.append(r_value ** 2)
                except:
                    scale_free_r2.append(0)
            else:
                scale_free_r2.append(0)
        else:
            scale_free_r2.append(0)
    
    # 选择R^2 > 0.8的最小power
    scale_free_r2 = np.array(scale_free_r2)
    best_power = None
    
    for i, (power, r2) in enumerate(zip(powers, scale_free_r2)):
        if r2 > 0.8:
            best_power = power
            break
    
    if best_power is None:
        best_power = powers[np.argmax(scale_free_r2)]
    
    return best_power


def calculate_tom(adj_matrix):
    """
    计算拓扑重叠矩阵（TOM）
    """
    n = adj_matrix.shape[0]
    tom = np.zeros((n, n))
    
    k = adj_matrix.sum(axis=1)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                tom[i, j] = 1.0
            else:
                shared_neighbors = (adj_matrix[i, :] * adj_matrix[j, :]).sum()
                min_k = min(k[i], k[j])
                denominator = min_k + 1 - adj_matrix[i, j]
                
                if denominator > 0:
                    tom[i, j] = (shared_neighbors + adj_matrix[i, j]) / denominator
                    tom[j, i] = tom[i, j]
                else:
                    tom[i, j] = 0
                    tom[j, i] = 0
    
    return tom


def construct_wgcna_network(expr_matrix, gene_names, soft_power=6, min_module_size=30):
    """
    构建WGCNA网络
    """
    print(f"    构建WGCNA网络 (soft power={soft_power})...")
    
    # 1. 计算相关性矩阵
    print("    计算相关性矩阵...")
    corr_matrix = calculate_correlation_matrix(expr_matrix, method='pearson')
    
    # 2. 计算邻接矩阵
    adj_matrix = np.abs(corr_matrix) ** soft_power
    
    # 3. 计算TOM
    print("    计算拓扑重叠矩阵...")
    tom_matrix = calculate_tom(adj_matrix)
    
    # 4. 层次聚类
    print("    层次聚类识别模块...")
    dissim_tom = 1 - tom_matrix
    dissim_tom[dissim_tom < 0] = 0
    
    # 转换为距离向量
    dist_vector = squareform(dissim_tom, checks=False)
    
    # 层次聚类
    linkage_matrix = linkage(dist_vector, method='average')
    
    # 动态切树
    n_genes = len(gene_names)
    max_clusters = max(2, n_genes // min_module_size)
    
    best_modules = None
    best_n_modules = 0
    
    for n_clusters in range(2, min(max_clusters + 1, 20)):
        module_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        unique_labels, counts = np.unique(module_labels, return_counts=True)
        valid_modules = counts >= min_module_size
        
        if valid_modules.sum() > best_n_modules:
            best_n_modules = valid_modules.sum()
            best_modules = module_labels
    
    if best_modules is None:
        best_modules = fcluster(linkage_matrix, 5, criterion='maxclust')
    
    # 5. 构建模块字典
    modules = {}
    for module_id in np.unique(best_modules):
        module_genes = [gene_names[i] for i in range(len(gene_names)) if best_modules[i] == module_id]
        
        if len(module_genes) >= min_module_size:
            modules[f'Module_{module_id}'] = module_genes
    
    print(f"    识别到 {len(modules)} 个模块")
    
    return modules, adj_matrix, tom_matrix


def calculate_module_preservation(modules1, modules2, expr1, expr2, gene_names):
    """
    计算模块保守性
    """
    preservation_results = []
    
    for module_name, module_genes in modules1.items():
        gene_indices = [i for i, g in enumerate(gene_names) if g in module_genes]
        
        if len(gene_indices) < 10:
            continue
        
        expr1_module = expr1[:, gene_indices]
        expr2_module = expr2[:, gene_indices]
        
        # 计算相关性矩阵
        corr1 = calculate_correlation_matrix(expr1_module, method='pearson')
        corr2 = calculate_correlation_matrix(expr2_module, method='pearson')
        
        # 计算保守性
        triu_idx = np.triu_indices(len(gene_indices), k=1)
        corr1_flat = corr1[triu_idx]
        corr2_flat = corr2[triu_idx]
        
        # 移除NaN
        mask = ~(np.isnan(corr1_flat) | np.isnan(corr2_flat))
        if mask.sum() < 10:
            continue
        
        corr1_clean = corr1_flat[mask]
        corr2_clean = corr2_flat[mask]
        
        try:
            preservation, pval = spearmanr(corr1_clean, corr2_clean)
            
            if not np.isnan(preservation):
                preservation_results.append({
                    'module': module_name,
                    'n_genes': len(gene_indices),
                    'preservation': preservation,
                    'pvalue': pval
                })
        except:
            continue
    
    return preservation_results
