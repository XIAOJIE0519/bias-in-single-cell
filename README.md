# Selection Bias in Single-Cell Analysis

This repository contains the code for analyzing selection bias in single-cell RNA-seq data across different studies.

## Overview

We systematically evaluate the comparability of "healthy control" samples from different studies through three-layer hypothesis testing:
- **H1: Composition Bias** - Cell type composition differences
- **H2: State Bias** - Cell state variations within cell types
- **H3: Network Bias** - Gene co-expression network differences (WGCNA)

## Core Files

### Main Scripts
- `main_analysis.py` - Complete analysis pipeline
- `run_final.py` - Quick analysis (H3 + differential expression only)
- `config.py` - Configuration parameters

### Utility Modules
- `utils_data_loading.py` - Data loading functions
- `utils_preprocessing.py` - Quality control and preprocessing
- `utils_annotation.py` - Cell type annotation
- `utils_hypothesis_testing.py` - H1 and H2 hypothesis testing
- `utils_wgcna.py` - WGCNA core algorithms
- `utils_network_analysis_improved.py` - H3 network analysis and differential expression
- `utils_result_tables.py` - Result table generation

## Requirements

```
Python 3.13
R 4.2.2

Python packages:
- scanpy
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- harmonypy
- statsmodels
```

## Usage

```bash
# Run complete analysis
python main_analysis.py

# Quick analysis (H3 + DE only)
python run_final.py
```

## Output

The analysis generates:
- 6 standardized result tables (Table_*.csv)
- Visualization figures
- Comprehensive analysis report

## Citation

If you use this code, please cite our paper:
[Paper information to be added]

## License

MIT License
