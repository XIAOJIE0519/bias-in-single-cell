# The illusion of the universal baseline in public single-cell controls

Authors: Zhichao Jiang2, Ximing Wang3, Shanjie Luan1,*

Correspondence to: Luan20050519@163.com

## Abstract

Public single-cell RNA sequencing studies increasingly reuse samples labelled as healthy, normal or control as if those labels defined a universal biological baseline [1-4]. This Critical Comment argues that the label itself is often a statistical fiction: it records a source-study decision, not proof that samples are exchangeable across donor populations, tissue handling, enrichment, modality and sequencing workflows. We use 28 public lung control samples from five studies as a bounded case study to illustrate the problem. A protocol-aware review separated whole-tissue single-cell datasets from enriched, sorted, single-nucleus and autopsy-derived datasets; integration diagnostics showed that corrected embeddings reduced visible study structure but did not certify compatibility; and donor-level sensitivity analyses still detected study-conditioned composition and cell-state differences [5-9]. The purpose of this evidence is not to introduce a new integration method or to assign every residual difference to biology. Instead, it shows why compatibility must be evaluated before public controls are pooled. We propose a transparency-first checklist: define the control label, report pre- and post-integration views, interpret batch-removal metrics alongside biological-conservation metrics, use donor-level rather than cell-level inference, and disclose unresolved metadata gaps. Integration can align representations, but it cannot turn incompatible studies into a shared baseline.

## Keywords

single-cell RNA sequencing; public controls; study compatibility; batch effects; integration diagnostics; donor-level analysis

## The illusion of the universal baseline

The most convenient assumption in public single-cell reuse is also one of the easiest to forget: a sample labelled healthy or control is treated as if it were a portable biological constant. In practice, that label is a decision made within a source study. It depends on who was eligible, how tissue was obtained, which cells or nuclei were captured, which chemistry was used, how the data were filtered, and what the original study needed the controls to control.
This distinction matters because modern integration tools can make incompatible datasets look reassuringly coherent [2,3,8,9]. A corrected embedding may reduce study separation, yet still leave open whether the underlying samples are comparable controls for disease analysis, atlas construction or model training. The central issue is therefore not whether integration is useful. It is useful. The issue is whether integration is being asked to answer a question it was not designed to answer: are these studies compatible enough to define a shared baseline?
We call this problem the illusion of the universal baseline. It is an illusion because a shared control label can conceal different sampling frames, protocols and residual confounders. It is universal only in appearance: once the source-study history is restored, the control label becomes conditional rather than absolute. The article therefore treats the lung-control case study as evidence for an interpretive practice: researchers should read control labels through their source-study histories before treating them as shared baselines.

## A lung-control case study exposes the problem

To make the argument concrete, we re-examined 28 public lung control samples from five GEO series. The full set was intentionally heterogeneous: it included whole-tissue scRNA-seq studies, lineage-negative enrichment, FACS-enriched fractions, and frozen rapid-autopsy single-nucleus data. We therefore treated the complete dataset as a diagnostic case rather than as an unbiased composition cohort. Only GSE173896 and GSE227691 were used as a controlled whole-tissue scRNA-seq subset for the most conservative comparisons (Figure 2A; Supplementary Table S1).
This design choice is a first-principles point, not a technical footnote. Before asking whether control samples differ, one must ask what kind of control each source study created. Composition claims require different compatibility than cell-state claims. Enriched epithelial or stromal fractions can be useful for state-level sensitivity analyses but are not interchangeable with whole-tissue composition samples. Single-nucleus autopsy data can be scientifically valuable while still being a poor match for fresh whole-cell composition inference.
The lung-control analysis therefore serves as a stress test for a common reuse habit. If samples that all carry a control label differ in source protocol, modality and donor context, then pooling them first and explaining differences later reverses the order of scientific reasoning. Compatibility should be an entry condition for pooling, not an afterthought after integration.

## Corrected embeddings are not evidence of compatibility

The corrected embedding is the most visually persuasive artifact in many public single-cell analyses. It can also be the most misleading if it is read as a certificate of exchangeability. In the lung-control case study, Harmony and scVI reduced study-associated structure compared with the unintegrated representation, but corrected embeddings did not remove the need for donor-level checks (Figure 2B-E; Supplementary Table S4) [8,9].
The same logic applies beyond this dataset. Integration methods optimize representations under assumptions and objectives; they do not reconstruct missing donor metadata, undo enrichment designs, or make an autopsy single-nucleus profile equivalent to a fresh whole-cell profile. A study can appear better mixed in embedding space while remaining unsuitable as a pooled control for a specific downstream question.
The operational punchline is simple: integration is not a substitute for study compatibility. A corrected UMAP can support visualization and exploratory annotation, but compatibility is established by the source-study design, transparent diagnostics and donor-level sensitivity analyses. Treating the embedding as biological fact creates a false sense of security and can propagate hidden incompatibility into disease comparisons.

## How to read integration metrics

Metrics such as average silhouette width (ASW), iLISI, cLISI and graph connectivity should be interpreted as a set of diagnostic questions rather than as a leaderboard [5]. ASW asks whether labels remain separated or overlap in a representation. When the label is study, lower batch ASW is consistent with weaker study separation. When the label is cell type, higher cell-type ASW can indicate better preservation of annotated biological structure.
iLISI asks a different question: how diverse are study labels within local neighborhoods? It behaves like a local diversity or entropy score for batch mixing, so a higher iLISI indicates more study mixing. cLISI applies the same logic to biological labels such as annotated cell types. A higher cLISI can therefore be a warning sign if integration is mixing cell types that should remain distinct. Graph connectivity checks whether cells carrying the same biological label remain connected rather than being fragmented across the representation.
These metrics were chosen because they expose the trade-off at the center of the critique. A method can improve batch mixing while also increasing biological over-mixing. The point is not to crown a best integration method, but to make visible the difference between reducing study structure and proving that studies are compatible controls (Figure 1C; Figure 2E; Supplementary Table S4). Detailed values belong in the Supplementary Materials so that the main text can focus on what the trends mean.

## A transparency-first checklist for public-control reuse

The practical response is a transparency-first workflow. First, define what the control label means in each source study, including donor eligibility, disease context, tissue region, modality, enrichment or sorting, and sequencing workflow. Second, build a study compatibility table before integration. This table should determine which datasets are eligible for composition analysis, which are suitable only for state-level sensitivity analysis, and which should remain diagnostic rather than inferential.
Third, report pre- and post-integration visualizations. The unintegrated view reveals study structure that a corrected embedding may hide; the integrated view shows what the correction changed. Fourth, pair batch-removal metrics with biological-conservation metrics. Reporting only batch mixing encourages overconfidence; reporting only cell-type preservation ignores the purpose of integration. Fifth, verify downstream conclusions at the donor or sample level. Individual cells are measurements, not independent donors.
Finally, researchers should include a buyer-beware statement when public controls remain imperfect. Missing age, sex, smoking status, tissue region or chemistry metadata should not be silently imputed or hidden by integration. Unresolved confounding is not a failure of the analysis if it is reported clearly; it becomes a failure when a pooled control baseline is presented as if those uncertainties did not exist.

## Beyond explicit batch removal

A constructive path forward is to reduce the pressure on integration to erase differences that should instead remain visible. Methods such as MapBatch illustrate conservative, batch-aware normalization strategies that aim to preserve biological signal rather than treating every batch-associated difference as noise [11]. The broader lesson is not that any one method should replace Harmony, scVI or other current tools. It is that analytical designs should preserve enough batch and protocol information for investigators to see when compatibility remains unresolved.
This perspective also changes how new tools should be judged. A method that produces a visually smooth embedding is not automatically better if it obscures incompatibility relevant to the scientific question. Conversely, a method that keeps some study structure visible may be valuable when that structure reflects real protocol or donor differences that must be reported. The field needs integration diagnostics that protect interpretability, not only aesthetics.

## Boundaries of the argument

This comment should not be read as an argument against public control reuse. Public single-cell data are indispensable, and carefully matched controls can increase power, improve annotation and support reproducible disease comparisons. The argument is narrower: control reuse should be conditional on study compatibility and donor-level validation.
The lung-control case study also has limits. It concerns one tissue context and relies on metadata available from public sources. Age, sex, smoking status, tissue region and chemistry were not uniformly available and were not imputed. Cell labels were marker-based and include low-confidence clusters, so fine subtype interpretation should remain cautious. The controlled subset contains only ten samples, which limits precision for donor-level inference. These boundaries are exactly why the article emphasizes transparency rather than a universal correction recipe.

## Conclusion

The healthy-control label is not a universal baseline. It is a study-conditioned label that must be interpreted through donor selection, protocol design, modality and metadata completeness. Integration can help align representations, but compatibility must be earned rather than assumed. Before public controls are pooled, researchers should show what integration changed, verify conclusions at the donor level, and state what remains unresolved.

## Figures and Supplementary Materials

Figure 1. From universal-baseline illusion to compatibility-first reuse. (A) A shared control label can hide incompatible donor and protocol histories. (B) Compatibility-first workflow for public-control reuse. (C) Conceptual guide to ASW, iLISI, cLISI and graph connectivity. (D) Buyer-beware checklist for transparent reuse of public single-cell controls.
Figure 2. Lung-control case study. (A) Study compatibility map for five public lung control datasets. (B-D) Unintegrated, Harmony-integrated and scVI representations coloured by source study. (E) Integration metrics showing why batch-removal and biological-conservation diagnostics must be read together. (F) Donor-level evidence summary showing that protocol-aware filtering reduced but did not eliminate study-conditioned heterogeneity.
Supplementary Materials. Supplementary Tables S1-S8 report dataset classification, quality control, marker filtering, annotation, integration metrics, composition diagnostics, pseudobulk analyses and sensitivity analyses. Supplementary Figures S1-S6 provide the all-study composition screen, pairwise pseudobulk DESeq2 heatmap and individual UMAP or metric panels [10].

## Data Availability Statement

The public data used in this study were obtained from the GEO series listed in Supplementary Table S1. The revised analysis outputs are summarized in Supplementary Tables S1-S8. The analysis code is available at https://github.com/XIAOJIE0519/bias-in-single-cell.

## Author Contributions

Ximing Wang: Original Writing, Conceptualization, Supervision.
Zhichao Jiang: Original Writing, Conceptualization, Methodology.
Shanjie Luan: Conceptualization, Methodology, Software, Writing and Editing, Visualization.

## Ethical approval

Not applicable.

## Funding

Not applicable.

## Declaration of interests

The authors declare no conflicts of interest.

## References
1. Luecken MD, Theis FJ. Current best practices in single-cell RNA-seq analysis: a tutorial. Mol Syst Biol. 2019;15:e8746. doi:10.15252/msb.20188746
2. Tran HTN, Ang KS, Chevrier M, Zhang X, Lee NYS, Goh M, et al. A benchmark of batch-effect correction methods for single-cell RNA sequencing data. Genome Biol. 2020;21:12. doi:10.1186/s13059-019-1850-9
3. Heumos L, Schaar AC, Lance C, Litinetskaya A, Drost F, Zappia L, et al. Best practices for single-cell analysis across modalities. Nat Rev Genet. 2023;24:550-572. doi:10.1038/s41576-023-00586-w
4. Li M, Zhang X, Ang KS, Ling J, Sethi R, Lee NYS, et al. DISCO: a database of Deeply Integrated human Single-Cell Omics data. Nucleic Acids Res. 2022;50:D596-D602. doi:10.1093/nar/gkab1020
5. Büttner M, Miao Z, Wolf FA, Teichmann SA, Theis FJ. A test metric for assessing single-cell RNA-seq batch correction. Nat Methods. 2019;16:43-49. doi:10.1038/s41592-018-0254-1
6. Squair JW, Gautier M, Kathe C, Anderson MA, James ND, Hutson TH, et al. Confronting false discoveries in single-cell differential expression. Nat Commun. 2021;12:5692. doi:10.1038/s41467-021-25960-2
7. Crowell HL, Soneson C, Germain PL, Calini D, Collin L, Raposo C, et al. muscat detects subpopulation-specific state transitions from multi-sample multi-condition single-cell transcriptomics data. Nat Commun. 2020;11:6077. doi:10.1038/s41467-020-19894-4
8. Korsunsky I, Millard N, Fan J, Slowikowski K, Zhang F, Wei K, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nat Methods. 2019;16:1289-1296. doi:10.1038/s41592-019-0619-0
9. Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative modeling for single-cell transcriptomics. Nat Methods. 2018;15:1053-1058. doi:10.1038/s41592-018-0229-2
10. Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biol. 2014;15:550. doi:10.1186/s13059-014-0550-8
11. Yong CH, Hoon S, de Mel S, Xu S, Scolnick JA, Huo D, Lovci MT, Chng WJ, Goh WWB. MapBatch: Conservative batch normalization for single cell RNA-sequencing data enables discovery of rare cell populations in a multiple myeloma cohort. Blood. 2021;138(Suppl 1):2954. doi:10.1182/blood-2021-150089