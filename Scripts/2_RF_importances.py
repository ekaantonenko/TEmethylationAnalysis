import argparse

import pandas as pd

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Random Forest feature importance analysis.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["regression_flanks", 'regression_TE'],
    required=True,
    help="Choose the mode: 'regression_flanks' or 'regression_TE'"
)
args = parser.parse_args()
mode = args.mode

########################################################################################
########################################################################################


# PARAMETERS
SEEDS = list(range(10))
OUT_PREFIX = f'../Results/2_{mode}_RF_importances'

df_feature_importances_mean, df_feature_importances_std = [], []

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

### Training the models (per window and target) on the whole dataset
for window_name, target_cols in windows_and_targets:

    for target_col in target_cols:
        if 'all' in target_col or 'TE' in target_col:
            X = df_cleaned.copy()

            if window_name == 'TE':
                X_window, y_window = prepare_X_y_te(X, target_col)
            else:
                X_window, y_window = prepare_X_y(X, target_col, window_name)

            X_window = X_window.drop(columns=['genome'])

            print(f"\nTarget: {target_col}")
            print(f'Number of samples for this target: {X_window.shape[0]}')
            print(f'Number of features for this target: {X_window.shape[1]}')

            importances_seeds = []
            ### Loop over different random seeds
            for RANDOM_STATE_model in SEEDS:

                model = create_model(X_window,
                                     mode=mode,
                                     modelname='RF',
                                     RANDOM_STATE_model=RANDOM_STATE_model)
                model.set_params(**best_params)

                ### Training
                model.fit(X_window, y_window)

                ### Storing feature importances
                feature_names = model.named_steps["preprocessor"].get_feature_names_out()
                final_model = model.named_steps["regressor"]
                importances_seeds.append(pd.Series(final_model.feature_importances_, index=feature_names))

            # Mean and standard deviation across seeds (one column per seed)
            importances_seeds = pd.concat(importances_seeds, axis=1)
            col = pd.MultiIndex.from_tuples([(window_name, target_col)], names=['Window', 'Target'])
            df_feature_importances_mean.append(pd.DataFrame(importances_seeds.mean(axis=1).values,
                                                            index=importances_seeds.index, columns=col))
            df_feature_importances_std.append(pd.DataFrame(importances_seeds.std(axis=1).values,
                                                           index=importances_seeds.index, columns=col))

df_feature_importances_mean = pd.concat(df_feature_importances_mean, axis=1)
df_feature_importances_mean.to_csv(f'{OUT_PREFIX}_mean.csv')
df_feature_importances_std = pd.concat(df_feature_importances_std, axis=1)
df_feature_importances_std.to_csv(f'{OUT_PREFIX}_std.csv')