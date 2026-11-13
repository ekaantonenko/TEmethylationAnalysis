import argparse

import pandas as pd

import shap

from sklearn.model_selection import GroupKFold

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run cross-validated SHAP feature importance analysis for the Random Forest models.")
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
N_FOLDS = 10
RANDOM_STATE_KFold = 42
RANDOM_STATE_model = 42
RANDOM_STATE_Sample = 42
SHAP_SAMPLE_FRAC = 1
OUT_PREFIX = f'../Results/3_SHAP/3_{mode}_SHAP_importances'

df_shap_importances_mean, df_shap_importances_std = [], []

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

# Cross-validation splitter (grouped by TE: a TE is never split between train and validation)
gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_KFold)

for window_name, target_cols in windows_and_targets:
    for target_col in target_cols:
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

        # Mean |SHAP| per feature, one row per fold
        shap_values_all = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_window, groups=groups)):
            X_train, X_val = X_window.iloc[train_idx], X_window.iloc[val_idx]
            y_train = y_window.iloc[train_idx]

            model = create_model(X_train,
                                 mode=mode,
                                 modelname='RF',
                                 RANDOM_STATE_model=RANDOM_STATE_model)
            model.set_params(**best_params)

            ### Training
            model.fit(X_train, y_train)

            # Transform validation set for SHAP
            X_val_transformed = model.named_steps['preprocessor'].transform(X_val)
            final_model = model.named_steps["regressor"]
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            # Sample subset of validation data for speed [by default, SHAP_SAMPLE_FRAC=1]
            X_val_sample = shap.utils.sample(X_val_transformed,
                                             int(len(X_val_transformed) * SHAP_SAMPLE_FRAC),
                                             random_state=RANDOM_STATE_Sample)

            explainer = shap.Explainer(final_model.predict, X_val_sample)
            shap_values = explainer(X_val_sample)
            shap_df = pd.DataFrame(shap_values.values, columns=feature_names)

            shap_fold_mean = shap_df.abs().mean().to_frame().T
            shap_fold_mean.index = [f'fold_{fold}']
            shap_values_all.append(shap_fold_mean)

        # Aggregate SHAP values across folds
        shap_folds_df = pd.concat(shap_values_all, axis=0)
        shap_folds_df.to_csv(f'{OUT_PREFIX}_{window_name}_{target_col}_perfold.csv')

        # Mean and standard deviation across folds
        col = pd.MultiIndex.from_tuples([(window_name, target_col)], names=['Window', 'Target'])
        df_shap_importances_mean.append(pd.DataFrame(shap_folds_df.mean(axis=0).values,
                                                     index=shap_folds_df.columns, columns=col))
        df_shap_importances_std.append(pd.DataFrame(shap_folds_df.std(axis=0).values,
                                                    index=shap_folds_df.columns, columns=col))

        # Checkpoint: rewrite the summary tables after each target, so partial results survive a crash
        pd.concat(df_shap_importances_mean, axis=1).to_csv(f'{OUT_PREFIX}_mean.csv')
        pd.concat(df_shap_importances_std, axis=1).to_csv(f'{OUT_PREFIX}_std.csv')