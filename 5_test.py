import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Random Forest models on the full dataset and evaluate them on the new cohort.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["regression_flanks", "regression_TE"],
    required=True,
    help="Choose the mode: 'regression_flanks' or 'regression_TE'"
)
parser.add_argument(
    "--data_filter",
    type=str,
    choices=["all", "all_nodup", "all_nodup_unseen", "poly_nodup", "poly_nodup_unseen"],
    required=True
)

args = parser.parse_args()
mode = args.mode
data_filter = args.data_filter

########################################################################################
########################################################################################

# PARAMETERS
RANDOM_STATE_model = 42
MIN_TEST_SAMPLES = 2           # pearsonr needs at least 2 points; smaller test sets are skipped
OUT_PREFIX = f'../Tables/5_{mode}_{data_filter}_TEST'

# Results containers
test_metrics = []              # One row of metrics per (window, target)
test_predictions = []

# Choosing the targets
if mode == 'regression_flanks':
    windows_and_targets = regression_flanks_windows_and_targets
    best_params = best_params_regression_flanks
else:  # 'regression_TE' (argparse already restricts the choices)
    windows_and_targets = regression_TE_windows_and_targets
    best_params = best_params_regression_TE


### import data ###
if mode == 'regression_TE':
    df_train = pd.read_csv('../Data/allTEs_upd_upd.csv')
    df_test = pd.read_csv('../Data/200_allTEs_upd_upd.csv')
else:
    df_train = pd.read_csv('../Data/allTEs_meth_upd_upd.csv')
    df_test = pd.read_csv('../Data/200_allTEs_meth_upd_upd.csv')

########################################################################################
########################################################################################

# Genomes present in both cohorts (removed from the test set by the 'nodup' filters)
common_genomes = [
    '2017','5279','6009','6017','6086','6145','6163','6911','6943','7058',
    '7067','7328','7350','7373','8259','8424','9078','9085','9100','9105',
    '9111','9133','9134','9433','9518','9533','9542','9543','9545','9549',
    '9557','9564','9567','9568','9588','9593','9594','9600','9616','9745',
    '9759','9762','9796','9832','9879','9908','9937'
]

if 'poly' in data_filter:
    df_train = df_train[df_train['TE.insertion_frequency_corr'] < 1]

if 'nodup' in data_filter:
    df_test = df_test[~df_test['genome'].astype(str).isin(common_genomes)]

if 'unseen' in data_filter:
    # Keep only test TEs whose (family, 5' gene, 3' gene) combination is absent from the training set
    print('Test TEs before filtering:', len(df_test))
    df_test = df_test.merge(
        df_train[['TE.family', '5p.Gene1.id', '3p.Gene1.id']].drop_duplicates(),
        on=['TE.family', '5p.Gene1.id', '3p.Gene1.id'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns='_merge')
    print('Test TEs after filtering:', len(df_test))


# Training loop: for each genomic window and target methylation feature
for window_name, target_cols in windows_and_targets:
    print(f"\n--- Test on new cohort for {window_name} window ---")

    for target_col in target_cols:
        df_test_target = df_test.dropna(subset=[target_col])

        if window_name == 'TE':
            X_train, y_train = prepare_X_y_te(df_train, target_col)
            X_test, y_test = prepare_X_y_te(df_test_target, target_col, filter=False)
        else:
            X_train, y_train = prepare_X_y(df_train, target_col, window_name)
            X_test, y_test = prepare_X_y(df_test_target, target_col, window_name, filter=False)

        X_test_genome = X_test['genome']
        X_train = X_train.drop(columns=['genome'])
        X_test = X_test.drop(columns=['genome'])

        ### IMPUTATION OF MISSING VALUES IN _TEST_ (using statistics from _TRAIN_)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
        X_test[num_cols] = SimpleImputer(strategy="median").fit(X_train[num_cols]).transform(X_test[num_cols])
        if len(cat_cols) > 0:
            X_test[cat_cols] = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols]).transform(X_test[cat_cols])

        print(f"\nTarget: {target_col}")
        print(f'Number of TRAIN samples for this target: {X_train.shape[0]}')
        print(f'Number of TEST samples for this target: {X_test.shape[0]}')
        print(f'Number of features for this target: {X_train.shape[1]}')

        if len(X_test) < MIN_TEST_SAMPLES:
            print(f'  Skipping {window_name} / {target_col}: not enough test samples after filtering')
            continue

        model = create_model(X_train,
                             mode=mode,
                             modelname='RF',
                             RANDOM_STATE_model=RANDOM_STATE_model)
        model.set_params(**best_params)

        # Training
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Store predictions
        test_predictions.append(pd.DataFrame({
                                    'Window': window_name,
                                    'Target': target_col,
                                    'TE': X_test.index,
                                    'genome': X_test_genome,
                                    'y_true': y_test.values,
                                    'y_pred': predictions
                                }))

        # Evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        # Pearson is undefined if either vector is constant
        if np.ptp(y_test.values) > 0 and np.ptp(predictions) > 0:
            pearson = pearsonr(y_test, predictions)[0]
        else:
            pearson = np.nan
        test_metrics.append({
                'Window': window_name,
                'Target': target_col,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'pearson': pearson
                })

### Store results ###
results_df = pd.DataFrame(test_metrics).set_index(['Window', 'Target'])
results_df.to_csv(f'{OUT_PREFIX}_results.csv')
all_predictions_df = pd.concat(test_predictions, ignore_index=True)
all_predictions_df.to_csv(f'{OUT_PREFIX}_all_predictions.csv', index=False)