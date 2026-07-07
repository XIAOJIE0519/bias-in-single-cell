# Data availability and input manifest

The input data are public GEO lung control samples. Raw `.h5` files are not included in this repository package. To rerun the full pipeline, download the corresponding matrices from GEO and place each file at the expected relative path shown below.

The repository includes the analysis code and output summaries needed to audit the manuscript results. The public data are listed here so that the analysis can be reconstructed without committing raw matrices to GitHub.

## Expected input layout

```text
GSE132771/
GSE159354/
GSE171524/
GSE173896/
GSE227691/
code/
result/
```

## Sample manifest

| Study | Sample ID | Expected file path | Analysis role |
|---|---|---|---|
| GSE132771 | GSM3891620 | `GSE132771/GSM3891620.h5` | diagnostic-only |
| GSE132771 | GSM3891621 | `GSE132771/GSM3891621.h5` | diagnostic-only |
| GSE132771 | GSM3891622 | `GSE132771/GSM3891622.h5` | diagnostic-only |
| GSE132771 | GSM3891623 | `GSE132771/GSM3891623.h5` | diagnostic-only |
| GSE132771 | GSM3891624 | `GSE132771/GSM3891624.h5` | diagnostic-only |
| GSE132771 | GSM3891625 | `GSE132771/GSM3891625.h5` | diagnostic-only |
| GSE159354 | GSM4827167 | `GSE159354/GSM4827167.h5` | diagnostic-only |
| GSE159354 | GSM4827184 | `GSE159354/GSM4827184.h5` | diagnostic-only |
| GSE159354 | GSM4827185 | `GSE159354/GSM4827185.h5` | diagnostic-only |
| GSE159354 | GSM4827199 | `GSE159354/GSM4827199.h5` | diagnostic-only |
| GSE159354 | GSM4827200 | `GSE159354/GSM4827200.h5` | diagnostic-only |
| GSE171524 | GSM5226574 | `GSE171524/GSM5226574.h5` | sensitivity-only |
| GSE171524 | GSM5226575 | `GSE171524/GSM5226575.h5` | sensitivity-only |
| GSE171524 | GSM5226576 | `GSE171524/GSM5226576.h5` | sensitivity-only |
| GSE171524 | GSM5226577 | `GSE171524/GSM5226577.h5` | sensitivity-only |
| GSE171524 | GSM5226578 | `GSE171524/GSM5226578.h5` | sensitivity-only |
| GSE171524 | GSM5226579 | `GSE171524/GSM5226579.h5` | sensitivity-only |
| GSE171524 | GSM5226580 | `GSE171524/GSM5226580.h5` | sensitivity-only |
| GSE173896 | GSM5282543 | `GSE173896/GSM5282543.h5` | controlled-primary |
| GSE173896 | GSM5282544 | `GSE173896/GSM5282544.h5` | controlled-primary |
| GSE173896 | GSM5282545 | `GSE173896/GSM5282545.h5` | controlled-primary |
| GSE173896 | GSM5282546 | `GSE173896/GSM5282546.h5` | controlled-primary |
| GSE173896 | GSM5282547 | `GSE173896/GSM5282547.h5` | controlled-primary |
| GSE173896 | GSM5282548 | `GSE173896/GSM5282548.h5` | controlled-primary |
| GSE227691 | GSM7105661 | `GSE227691/GSM7105661.h5` | controlled-primary |
| GSE227691 | GSM7105662 | `GSE227691/GSM7105662.h5` | controlled-primary |
| GSE227691 | GSM7105663 | `GSE227691/GSM7105663.h5` | controlled-primary |
| GSE227691 | GSM7105664 | `GSE227691/GSM7105664.h5` | controlled-primary |

## Metadata boundary

Age, sex, smoking status, tissue region and chemistry were not uniformly available across source studies and were not imputed. This is an intentional boundary of the manuscript argument and is part of the reason the repository separates controlled-primary, diagnostic-only and sensitivity-only datasets.

