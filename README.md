# TEmethylationAnalysis

In this repository, we evaluate which features are the most predictive for the methylation of Transposable Elements (TEs) in ***A. thaliana***, using Random Forest regression.

> Paper link

## Repository structure

- [Scripts/](Scripts): code used to run the experiments.
- [Results/](Results): results of the experiments.
- [Data/](Data): data used in the experiments.

## Requirements

```bash
pip install pandas numpy scipy "scikit-learn>=1.6" shap PyALE
```

## Usage

Run the scripts from `Scripts/`, since all paths are relative to it. Every script takes a `--mode` argument:

- `regression_TE`: predict the methylation of the TE itself.
- `regression_flanks`: predict the methylation of the regions flanking the TE.

```bash
cd Scripts
python 1_crossvalidation.py --mode regression_TE --data_filter all --modelname RF
```

## Experiments

| Script | Description | Extra arguments |
|---|---|---|
| `1_crossvalidation.py` | 10-fold grouped cross-validation: metrics per fold, averaged metrics and predictions. | `--data_filter`, `--modelname` |
| `2_RF_importances.py` | Random Forest feature importances on the full dataset (mean and std over 10 seeds). | – |
| `3_SHAP_importances.py` | SHAP importances in 10-fold grouped cross-validation (per fold, mean and std). | – |
| `4_ALE_importances.py` | ALE curves of the numeric features on a grouped 80/20 train/validation split. | – |
| `5_test.py` | Training on the full dataset and evaluation on an independent cohort. | `--data_filter` |
| `6_SHAP_importances_on_test.py` | SHAP importances on the independent cohort. | – |
| `7_ALE_importances_on_test.py` | ALE curves of all features on the independent cohort. | – |

#### `--data_filter` in `1_crossvalidation.py`

- `all`: all TEs; folds are grouped by TE.
- `poly`: polymorphic TEs only (insertion frequency < 1); folds are grouped by TE.
- `bygenome`: folds are grouped by genome, so test genomes are unseen during training.
- `bygenome-seen`: as `bygenome`, keeping only TEs present in both the training and the test fold.
- `bygenome-fixed`: currently identical to `bygenome` (its filter is disabled).

#### `--modelname` is `RF` (default), `linear` or `mean-genome`.

#### `--data_filter` in `5_test_new_cohort.py`

The value combines the following options (`all`, `all_nodup`, `all_nodup_unseen`, `poly_nodup`, `poly_nodup_unseen`):

- `all` / `poly`: train on all TEs, or on polymorphic TEs only.
- `nodup`: remove from the test cohort the genomes also present in the training cohort.
- `unseen`: remove test TEs whose family and 5′/3′ flanking genes occur in the training set.

Scripts 6 and 7 always use the `all_nodup_unseen` filtering.

## Outputs

Results are saved as CSV tables prefixed with the script number (`1_…` to `7_…`). Tables whose columns are labelled by window and target have two header rows; read them with:

```python
pd.read_csv(path, header=[0, 1], index_col=0)
```
