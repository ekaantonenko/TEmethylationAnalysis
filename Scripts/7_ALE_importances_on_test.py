import os
import argparse

import pandas as pd
import numpy as np

from PyALE import ale
from sklearn.impute import SimpleImputer

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run ALE analysis of the Random Forest models on the new-cohort test set.")
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
OUT_DIR = '../Results/7_ALE/test'

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
    df_test  = pd.read_csv('../Data/200_allTEs_upd_upd.csv')
else:
    df_train = pd.read_csv('../Data/allTEs_meth_upd_upd.csv')
    df_test  = pd.read_csv('../Data/200_allTEs_meth_upd_upd.csv')

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
df_test = df_test.merge(
    df_train[['TE.family', '5p.Gene1.id', '3p.Gene1.id']].drop_duplicates(),
    on=['TE.family', '5p.Gene1.id', '3p.Gene1.id'],
    how='left',
    indicator=True
).query('_merge == "left_only"').drop(columns='_merge')

os.makedirs(OUT_DIR, exist_ok=True)

########################################################################################
########################################################################################

def passthrough_encode(df):
    # PyALE converts the feature to an ordered pd.Categorical of strings internally;
    # turn it back into plain object dtype so the pipeline's OrdinalEncoder sees
    # the same kind of values it was fitted on.
    return df.astype(str).astype(object)


for window_name, target_cols in windows_and_targets:
    for target_col in target_cols:
        if not ('all' in target_col or 'TE' in target_col):
            continue

        print(f"\n{'='*60}")
        print(f"Target: {target_col} | Window: {window_name}")

        # --- Prepare train and test ---
        df_test_target = df_test.dropna(subset=[target_col])

        if window_name == 'TE':
            X_train, y_train = prepare_X_y_te(df_train, target_col)
            X_test, y_test = prepare_X_y_te(df_test_target, target_col, filter=False)
        else:
            X_train, y_train = prepare_X_y(df_train, target_col, window_name)
            X_test, y_test = prepare_X_y(df_test_target, target_col, window_name, filter=False)

        X_train = X_train.drop(columns=['genome'])
        X_test = X_test.drop(columns=['genome'])

        # --- Impute test using train statistics ---
        # Note: ALE needs the original feature space (not preprocessor-transformed),
        # so we impute X_test directly and pass it to ale() with the full pipeline
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

        print(f"Train samples : {X_train.shape[0]}")
        print(f"Test samples  : {X_test.shape[0]}")
        print(f"Features      : {X_train.shape[1]}")

        if X_test.empty:
            print(f'  Skipping {window_name} / {target_col}: no test samples after filtering')
            continue

        # --- Train model on full train set ---
        model = create_model(X_train,
                             mode=mode,
                             modelname='RF',
                             RANDOM_STATE_model=RANDOM_STATE_model)
        model.set_params(**best_params)
        model.fit(X_train, y_train)

        # Global baseline: mean prediction over full training set
        baseline = np.mean(model.predict(X_test))

        # Get numeric feature names from preprocessor
        preprocessor = model.named_steps['preprocessor']
        num_cols_for_model = preprocessor.named_transformers_['num'].get_feature_names_out()

        for feature in X_test.columns:

            # Skip features with insufficient unique values for ALE
            if X_test[feature].nunique() < 2:
                print(f"    Skipping {feature}: <2 unique values in the test set.")
                continue

            try:
                if feature in num_cols_for_model:
                    ale_res = ale(
                        X=X_test,
                        model=model,
                        feature=[feature],
                        include_CI=True,
                        plot=False
                    )
                else:
                    ale_res = ale(
                        X=X_test,
                        model=model,
                        feature=[feature],
                        feature_type='discrete',
                        encode_fun=passthrough_encode,
                        predictors=list(X_train.columns),
                        include_CI=True,
                        plot=False
                    )

                ale_res['effect_unscaled']  = baseline + ale_res['eff']
                ale_res['ci_low_abs']       = baseline + ale_res['lowerCI_95%']
                ale_res['ci_high_abs']      = baseline + ale_res['upperCI_95%']
                ale_res['feature']          = feature
                ale_res['target']           = target_col
                ale_res['window']           = window_name

                out_path = os.path.join(OUT_DIR, f'ALE_{mode}_{window_name}_{target_col}_{feature}.csv')
                ale_res.to_csv(out_path)

            except Exception as e:
                print(f"    ALE failed for {feature}: {e}")
                continue

print("\nDone.")