# Credit Portfolio Segmentation and Risk Pattern Analysis

Unsupervised learning project (GUVI x HCL) that groups loan borrowers by financial and credit profile, then compares historical default rates across the groups. The results are presented in an interactive Streamlit dashboard.

**Live dashboard:** _add your Streamlit link here after deploying_

## Problem statement
Financial institutions deal with borrowers who have different incomes, credit scores, employment histories and loan needs. This project uses clustering to find distinct borrower segments, profiles each segment, and compares historical default rates between segments. The `Default` column is **never** used to form the clusters. It is used only afterwards to compare outcomes.

## Dataset
- **Source:** `loans_granting_model_dataset.csv` (Lending Club loan data), provided with the GUVI project brief.
- **Size:** 1,347,681 loans, 15 columns, issued 2007-2018.
- **Used columns:** annual income (`revenue`), debt-to-income ratio (`dti_n`), loan amount, FICO score, employment length, loan purpose, home ownership, state, issue date, and `Default`.
- **Dropped:** `title`, `desc` (free text, mostly missing), `zip_code`, and `experience_c` (constant).
- The raw file is about 160 MB, so it is **not included** in this repository. Download it from the project brief and place it at `data/raw/loans_granting_model_dataset.csv`.

## Project structure
```
credit-portfolio-segmentation/
├── app/
│   ├── app.py                 # Streamlit dashboard
│   └── data/                  # small summary files used by the dashboard
├── data/
│   ├── raw/                   # raw CSV (not tracked)
│   └── processed/             # cleaned and clustered data (not tracked)
├── models/                    # saved scaler, PCA and K-Means (not tracked)
├── notebooks/
│   ├── 01_cleaning_eda.ipynb
│   ├── 02_features_pca_kmeans.ipynb
│   └── 03_dimred_hierarchical_dbscan.ipynb
├── reports/figures/           # saved plots
├── src/
│   ├── config.py              # paths, random seed, constants
│   └── preprocessing.py       # cleaning and feature functions
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/ssrikanthssk/credit-portfolio-segmentation.git
cd credit-portfolio-segmentation
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
Place the raw CSV in `data/raw/`, then run the notebooks in order (01, 02, 03). The last cell of notebook 03 saves the files the dashboard needs into `app/data/`. Start the dashboard with:
```bash
streamlit run app/app.py
```

## Methodology
1. **Cleaning:** parsed `issue_d` into dates (year and month features), converted employment length to years (the "NI" value, meaning no information, was filled with the median and flagged), capped income and DTI at the 99th percentile, log-transformed income, and grouped rare loan purposes. There were no duplicate rows and no missing values in the numeric columns.
2. **EDA:** distributions, correlations, and default rate by purpose, state, home ownership and year (overall default rate: 20.0%).
3. **Scaling and PCA:** the five numeric financial features (`revenue_log`, `dti_n`, `loan_amnt`, `fico_n`, `emp_length_years`) were standardized. PCA with 4 components keeps 91.3% of the variance.
4. **Clustering:** K-Means (k = 4) on all 1.3M borrowers; hierarchical clustering (Ward linkage) and DBSCAN on a stratified 10,000-row sample.
5. **Dimensionality reduction for visualization:** PCA, t-SNE and UMAP on the 10,000-row sample.
6. **Profiling and default analysis:** cluster averages, purpose and home ownership mix per cluster, and default rate per cluster, state and issue year.
7. **Anomaly detection:** DBSCAN noise points (`eps = 0.8`, `min_samples = 10`).

**Design decision:** a first K-Means run that included the one-hot encoded purpose and home ownership columns produced clusters that were simply "own / rent / mortgage / home improvement". The final clustering therefore uses only the numeric financial features. Purpose and home ownership are used afterwards to describe each cluster.

## Results

### K-Means segments (full data)
| Segment | Borrowers | Avg income | Avg DTI | Avg loan | Avg FICO | Default rate |
|---|---|---|---|---|---|---|
| Prime, low risk | 170,907 | $78,111 | 14.4 | $13,481 | 759 | **9.6%** |
| High income, large loans | 301,605 | $123,849 | 17.3 | $25,885 | 697 | 21.3% |
| Established mid-income | 429,593 | $65,554 | 19.6 | $11,653 | 687 | 21.1% |
| Early career, small loans | 445,576 | $52,142 | 18.8 | $9,653 | 687 | 22.0% |

### Algorithm comparison (10,000-row sample)
| Algorithm | Clusters | Noise | Silhouette |
|---|---|---|---|
| K-Means | 4 | 0% | 0.218 |
| Hierarchical (Ward) | 4 | 0% | 0.176 |
| DBSCAN (eps = 0.8) | 1 core cluster | 1.7% | not defined |

### Key findings
- One segment (high FICO, low DTI) has about half the overall default rate. The other three segments are close to each other (21-22%), so credit score and DTI separate risk better than income or loan size.
- Silhouette scores are low (about 0.2) and the 2D projections overlap: borrower characteristics form a continuum, not sharply separated groups.
- K-Means and hierarchical clustering agree on a small low-risk prime group.
- DBSCAN found one dense core, so it is used for anomaly detection. The 173 anomalies in the sample are unusual but not risky: they have very high FICO scores (about 779), higher income and larger loans, and a 14% default rate.

## Limitations
- Low silhouette scores mean segment boundaries are soft.
- t-SNE, UMAP, hierarchical clustering and DBSCAN were run on a 10,000-row sample, not the full data.
- Segments describe groups of borrowers, not individuals. Cluster membership should not be used to draw conclusions about a single borrower.
- Historical default rates do not guarantee future outcomes.

## Reproducibility
Random seed `42` is set in `src/config.py` and used by all models and samples.

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, UMAP, SciPy, Plotly, Streamlit, Git and GitHub.

## Author
Srikanth