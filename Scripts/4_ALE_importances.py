import argparse

import pandas as pd
import numpy as np

from PyALE import ale

from sklearn.model_selection import GroupShuffleSplit

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute ALE (accumulated local effects) curves for the Random Forest models.")
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
RANDOM_STATE_TTsplit = 42
RANDOM_STATE_model = 42
OUT_PREFIX = f'../Results/4_ALE/4_ALE_{mode}'

# Choosing the targets
if mode == 'regression_flanks':
    windows_and_targets = regression_flanks_windows_and_targets
    best_params = best_params_regression_flanks
else:  # 'regression_TE' (argparse already restricts the choices)
    windows_and_targets = regression_TE_windows_and_targets
    best_params = best_params_regression_TE


### import data ###
if mode == 'regression_TE':
    df_cleaned = pd.read_csv('../Data/allTEs_upd_upd.csv')
else:
    df_cleaned = pd.read_csv('../Data/allTEs_meth_upd_upd.csv')

########################################################################################
########################################################################################

# Train/validation splitter (grouped by TE: a TE is never split between train and validation)
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE_TTsplit)

for window_name, target_cols in windows_and_targets:

    for target_col in target_cols:
        if 'all' in target_col or 'TE' in target_col:
            X = df_cleaned.copy()

            if window_name == 'TE':
                X_window, y_window = prepare_X_y_te(X, target_col)
            else:
                X_window, y_window = prepare_X_y(X, target_col, window_name)
            X_window = X_window.drop(columns=['genome'])

            groups = X_window.index

            print(f"\nWindow: {window_name}, Target: {target_col}")
            print(f'Number of samples for this target: {X_window.shape[0]}')
            print(f'Number of features for this target: {X_window.shape[1]}')

            # Single grouped train/validation split
            train_idx, val_idx = next(gss.split(X_window, groups=groups))
            X_train, X_val = X_window.iloc[train_idx], X_window.iloc[val_idx]
            y_train = y_window.iloc[train_idx]

            model = create_model(X_train,
                                 mode=mode,
                                 modelname='RF',
                                 RANDOM_STATE_model=RANDOM_STATE_model)
            model.set_params(**best_params)

            ### Training
            model.fit(X_train, y_train)

            # Mean prediction, used to put the centred ALE effects back on the target scale
            baseline = np.mean(model.predict(X_val))

            preprocessor = model.named_steps['preprocessor']
            num_cols_for_model = preprocessor.named_transformers_['num'].get_feature_names_out()

            for feature in num_cols_for_model:
                ale_res = ale(X=X_val,
                              model=model,
                              feature=[feature],
                              include_CI=True,  # add 95% confidence intervals
                              plot=False)       # return the table only, no plot

                ale_res['effect_unscaled'] = baseline + ale_res['eff']
                ale_res["ci_low_abs"] = baseline + ale_res["lowerCI_95%"]
                ale_res["ci_high_abs"] = baseline + ale_res["upperCI_95%"]

                ale_res.to_csv(f'{OUT_PREFIX}_{window_name}_{target_col}_{feature}.csv')