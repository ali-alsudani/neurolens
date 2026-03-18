# NeuroLens

**Classifying Psychiatric Disorders from Postmortem Brain Gene Expression**

NeuroLens is a machine learning pipeline that classifies schizophrenia (SCZ), bipolar disorder (BD), major depressive disorder (MDD), and healthy controls (CTL) using postmortem brain gene expression data. The project integrates 9 public GEO datasets (607 samples, 10,678 Affymetrix gene probes, 471 unique patients) and applies SHAP-based stable feature selection to identify 100 disease-relevant probes.

## Key Results

| Model | F1 Macro (5-Fold CV) | Probes |
|-------|---------------------|--------|
| Dummy Baseline | 0.136 | - |
| Logistic Regression | 0.550 | 100 probes |
| XGBoost | 0.716 | 100 probes |

Biological validation confirmed 7 of the top 20 SHAP-selected genes in published psychiatric genetics literature, including NPTX2 (validated SCZ biomarker), TREM2 (neuroinflammation), HLA-DRB1 (SCZ GWAS risk gene), and FZD6 (depression via Wnt pathway).

## Project Structure

```
neurolens/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── EDA-VB.ipynb              # Exploratory data analysis (Version B)
│   ├── Model-VB.ipynb            # Modeling, SHAP, evaluation (Version B)
│   ├── 04_GAN_augmentation.ipynb # WGAN-GP augmentation experiments
│   ├── EDA-VA.ipynb              # EDA (Version A, PFC only)
│   └── Model-VA.ipynb            # Modeling (Version A, future work)
├── data/
│   ├── raw/                      # GEO series matrix files (not tracked)
│   ├── processed/                # ML-ready CSV files (not tracked)
│   ├── stable_probes_100.csv     # 100 consensus probes (5/5 folds)
│   ├── stable_probes_758.csv     # 758 probes (3+ folds)
│   └── stable_probes_783.csv     # 783 probes (alternative threshold)
├── figures/                      # All generated plots
├── models/                       # Saved model files
├── src/                          # Source code (future refactoring)
└── docs/                         # Documentation
```

## Datasets

All datasets are publicly available from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/).

| Dataset | Platform | Brain Region | Samples | Classes |
|---------|----------|-------------|---------|---------|
| GSE53987 | GPL570 | PFC, HPC, STR | 205 | SCZ, BD, MDD, CTL |
| GSE21138 | GPL570 | PFC (BA46) | 59 | SCZ, CTL |
| GSE17612 | GPL570 | PFC (BA10) | 51 | SCZ, CTL |
| GSE21935 | GPL570 | BA22 | 42 | SCZ, CTL |
| GSE54568 | GPL570 | PFC (BA9) | 30 | MDD, CTL |
| GSE12649 | GPL96 | PFC (BA46) | 102 | SCZ, BD, CTL |
| GSE5388/5389 | GPL96 | PFC / OFC | 82 | BD, CTL |
| SMRI Altar C | GPL96 | PFC (BA46) | 44 | SCZ, BD, MDD, CTL |

**Total after QC:** 607 samples, 10,678 shared probes, 471 unique patients

## Methodology

### Preprocessing
1. **Cross-platform probe matching:** Keep only probes present on both GPL570 and GPL96 platforms
2. **Log2 transformation:** Normalize raw intensities
3. **ComBat batch correction:** Remove lab-specific effects while preserving diagnosis signal (Johnson et al. 2007, Biostatistics). PC1 dropped from 90.8% to 13.5% after correction.
4. **Outlier removal:** PCA-based detection, 8 samples removed (>3 SD)
5. **Variance filtering:** 1,187 low-variance probes removed (bottom 10th percentile)

### Feature Selection
SHAP-based stable consensus methodology (Parvandeh et al. 2020, Bioinformatics):
1. Train XGBoost in each of 5 CV folds
2. Compute SHAP values on training data only (no leakage)
3. Rank probes by mean |SHAP| per fold
4. Threshold: probes above mean SHAP value (data-driven)
5. Consensus: keep probes selected in all 5 folds = **100 stable probes**

### Evaluation
- **Metric:** F1 Macro (treats all 4 classes equally)
- **Cross-validation:** StratifiedGroupKFold (5 folds)
- **GroupKFold:** Prevents data leakage from patients with multiple brain region samples (GSE53987)

## Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/AliAlSudani0/psychiatric-multi-classifier-project.git


# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Download

Raw GEO data files are not included in the repository due to size. To download:

1. Visit each GEO accession page (GSE53987, GSE21138, GSE17612, GSE21935, GSE54568, GSE12649, GSE5388, GSE5389)
2. Download the series matrix files
3. Place them in `data/raw/`
4. The SMRI Altar C dataset requires access through the Stanley Medical Research Institute

Alternatively, run the preprocessing notebooks which include download instructions.

## Usage

### Run the full pipeline

```bash
# 1. Preprocessing and EDA
jupyter notebook notebooks/EDA-VB.ipynb

# 2. Modeling and evaluation
jupyter notebook notebooks/Model-VB.ipynb

# 3. (Optional) GAN augmentation experiments
jupyter notebook notebooks/04_GAN_augmentation.ipynb
```

### Reproduce results

The processed data files in `data/processed/` and stable probe lists in `data/` allow you to skip preprocessing and run modeling directly:

```python
import pandas as pd

# Load ML-ready data
df = pd.read_csv('data/processed/ml_ready_max_samples.csv')

# Load 100 stable SHAP probes
stable_probes = pd.read_csv('data/stable_probes_100.csv')['0'].tolist()

# Ready for modeling
X = df[stable_probes]
```

## Key Findings

### Model Performance
- 100 stables feature selection improved F1 from baseline levels to 0.716 (XGBoost)
- 100 probes selected by consensus across all 5 CV folds outperform all 10,678 probes
- Feature selection contributes more to performance than model complexity

### Biological Validation
7 of the top 20 SHAP-selected genes confirmed in published psychiatric genetics:

| Gene | Rank | Evidence |
|------|------|----------|
| NPTX2 | #1 | Validated SCZ biomarker (Manchia 2017, Bhatt 2021) |
| S100A8 | #2 | Upregulated in SCZ brain and blood |
| TREM2 | #11 | Microglia activation, neuroinflammation |
| HLA-DRB1 | #13 | SCZ GWAS risk gene |
| FZD6 | #15 | Depression via Wnt pathway (CRISPR validated) |
| ERC1 | #16 | Synaptic scaffold, deletions cause psychiatric symptoms |
| PSPH | #18 | Serine synthesis for NMDA receptor, elevated in SCZ |

### Disorder-Specific Signatures
Each disorder is driven by distinct gene pathways:
- **BD:** Immune/inflammatory genes (SAMSN1, PTX3, HLA-DRB1)
- **CTL:** Synaptic markers (CDKL1, NPTX2, CAMK2A)
- **MDD:** Wnt signaling + circadian rhythm (FZD6, NR1D2)
- **SCZ:** Neuroinflammation (TREM2, S100A8, HES2)

## Limitations

- **Sample size:** 607 samples limits generalization. GAN augmentation (WGAN-GP) was attempted but did not improve results.
- **Postmortem tissue:** RNA degradation, cell type mixing, and stress gene activation affect measurements.
- **Brain region confound:** PC1 captures brain region (35.4% variance, F=1269), not disease. Covariates not regressed before training.
- **Confound genes:** 10 of 20 top SHAP genes lack published psychiatric evidence and are potential confounds.

## Future Work

- **Covariate regression:** Remove brain region, age, and sex effects before training (Gandal 2018, Jaffe 2015)
- **Version A (PFC only):** 414 samples, 20,049 probes. Eliminates brain region confounder. Data ready.
- **Blood-based validation:** Test if SHAP-selected genes are detectable in blood from living patients
- **Multi-modal integration:** Combine gene expression + brain imaging + clinical scores

## References

- Gandal et al. (2018). Shared molecular neuropathology across major psychiatric disorders. *Science*, 359(6376), 693-697.
- Hirschfeld et al. (2003). Bipolar disorder misdiagnosis. *AJMC*.
- Jaffe et al. (2015). Developmental and genetic regulation of the human cortex transcriptome. *Nature Neuroscience*, 18(7), 1081-1089.
- Johnson et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Parvandeh et al. (2020). Consensus features nested cross-validation. *Bioinformatics*, 36(10), 3093-3098.
- Singh & Rajput (2006). Misdiagnosis of bipolar disorder. *Psychiatry (Edgmont)*, 3(10), 57-63.

## Tools and Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, imbalanced-learn, matplotlib, ComBat (neuroCombat), PyTorch (WGAN-GP), Jupyter Notebook

## Author

**Ali Alsudani**
MSc Neuroscience | Data Science Bootcamp | SPICED Academy Berlin
March 2026

## License

See [LICENSE](LICENSE) for details.
