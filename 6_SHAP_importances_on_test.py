import os
import argparse

import pandas as pd
import numpy as np

import shap

from sklearn.impute import SimpleImputer

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run SHAP analysis of the Random Forest models on the new-cohort test set.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["regression_flanks", "regression_TE"],
    required=True,
    help="Choose the mode: 'regression_flanks' or 'regression_TE'"
)
args = parser.parse_args()
mode = args.mode

########################################################################################
########################################################################################

# PARAMETERS
RANDOM_STATE_model = 42
RANDOM_STATE_Sample = 42
SHAP_SAMPLE_FRAC = 1
OUT_PATH = f'../Tables_update/6_SHAP/6_{mode}_SHAP_on_test.csv'

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

# Test-set filtering: same as data_filter='all_nodup_unseen' in the new-cohort test script

# Genomes present in both cohorts (removed from the test set)
common_genomes = [
    '2017','5279','6009','6017','6086','6145','6163','6911','6943','7058',
    '7067','7328','7350','7373','8259','8424','9078','9085','9100','9105',
    '9111','9133','9134','9433','9518','9533','9542','9543','9545','9549',
    '9557','9564','9567','9568','9588','9593','9594','9600','9616','9745',
    '9759','9762','9796','9832','9879','9908','9937'
]

df_test = df_test[~df_test['genome'].astype(str).isin(common_genomes)]

# Keep only test TEs whose (family, 5' gene, 3' gene) combination is absent from the training set
print('Test TEs before filtering:', len(df_test))
df_test = df_test.merge(
    df_train[['TE.family', '5p.Gene1.id', '3p.Gene1.id']].drop_duplicates(),
    on=['TE.family', '5p.Gene1.id', '3p.Gene1.id'],
    how='left',
    indicator=True
).query('_merge == "left_only"').drop(columns='_merge')
print('Test TEs after filtering:', len(df_test))

########################################################################################
########################################################################################

# Collect all results across targets into one long dataframe
all_results = []

for window_name, target_cols in windows_and_targets:

    for target_col in target_cols:

        print(f"\n{'='*60}")
        print(f"Target: {target_col} | Window: {window_name}")

        df_test_target = df_test.dropna(subset=[target_col])

        if window_name == 'TE':
            X_train, y_train = prepare_X_y_te(df_train, target_col)
            X_test, y_test = prepare_X_y_te(df_test_target, target_col, filter=False)
        else:
            X_train, y_train = prepare_X_y(df_train, target_col, window_name)
            X_test, y_test = prepare_X_y(df_test_target, target_col, window_name, filter=False)

        X_train = X_train.drop(columns=['genome'])
        X_test = X_test.drop(columns=['genome'])

        ### IMPUTATION OF MISSING VALUES IN _TEST_ (using statistics from _TRAIN_)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
        X_test[num_cols] = (
            SimpleImputer(strategy="median")
            .fit(X_train[num_cols])
            .transform(X_test[num_cols])
        )
        if len(cat_cols) > 0:
            X_test[cat_cols] = (
                SimpleImputer(strategy="most_frequent")
                .fit(X_train[cat_cols])
                .transform(X_test[cat_cols])
            )

        print(f'Number of TRAIN samples for this target: {X_train.shape[0]}')
        print(f'Number of TEST samples for this target: {X_test.shape[0]}')
        print(f'Number of features for this target: {X_train.shape[1]}')

        if X_test.empty:
            print(f'  Skipping {window_name} / {target_col}: no test samples after filtering')
            continue

        model = create_model(X_train,
                             mode=mode,
                             modelname='RF',
                             RANDOM_STATE_model=RANDOM_STATE_model)
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        # Transform test set for SHAP
        X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        final_model = model.named_steps["regressor"]

        # Sample subset of test data for speed [by default, SHAP_SAMPLE_FRAC=1]
        X_test_sample = shap.utils.sample(
            X_test_transformed,
            int(len(X_test_transformed) * SHAP_SAMPLE_FRAC),
            random_state=RANDOM_STATE_Sample
        )

        explainer = shap.Explainer(final_model.predict, X_test_sample)
        shap_values = explainer(X_test_sample)

        # Mean and std of |SHAP| across test samples
        abs_shap = np.abs(shap_values.values)

        all_results.append(pd.DataFrame({
            'feature'       : feature_names,
            'shap_mean_abs' : abs_shap.mean(axis=0),
            'shap_std_abs'  : abs_shap.std(axis=0),
            'target'        : target_col,
            'window'        : window_name
        }))

########################################################################################

out_df = pd.concat(all_results, axis=0, ignore_index=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved results to {OUT_PATH}")
print(out_df.head(20).to_string(index=False))