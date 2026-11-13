import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

### import local ###
from _prepareData import prepare_X_y, prepare_X_y_te
from _features_and_targets import regression_flanks_windows_and_targets, regression_TE_windows_and_targets
from _model import create_model
from _best_params import best_params_regression_flanks, best_params_regression_TE

########################################################################################
########################################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run grouped K-fold cross-validation of TE methylation regression models.")
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
    choices=['all', 'bygenome', 'bygenome-seen', 'poly'],
    required=True
)

parser.add_argument(
    "--modelname",
    type=str,
    choices=["RF", "linear", "mean-genome"],
    default="RF",
    help="Choose the modelname: 'RF', 'linear', or 'mean-genome'"
)

args = parser.parse_args()
mode = args.mode
modelname = args.modelname
data_filter = args.data_filter

########################################################################################
########################################################################################

# PARAMETERS
N_FOLDS = 10
RANDOM_STATE_KFold = 42
RANDOM_STATE_model = 42
OUT_PREFIX = f'../Tables/1_{mode}_{modelname}_{data_filter}_CV'


# Defining the dictionaries
evaluation_results = {}
fold_results = {}              # Stores all fold-level results
fold_predictions = []

# Choosing the targets
if mode == 'regression_flanks':
    windows_and_targets = regression_flanks_windows_and_targets
    best_params = best_params_regression_flanks
else:   # 'regression_TE'
    windows_and_targets = regression_TE_windows_and_targets
    best_params = best_params_regression_TE


### import data ###
if mode == 'regression_TE':
    df_cleaned = pd.read_csv('../Data/allTEs_upd_upd.csv')
else:
    df_cleaned = pd.read_csv('../Data/allTEs_meth_upd_upd.csv')
    

if data_filter == "poly":
    df_cleaned = df_cleaned[df_cleaned['TE.insertion_frequency_corr'] < 1]


########################################################################################
########################################################################################

# Cross-validation splitter
gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_KFold)


# Training loop: for each genomic window and target methylation feature
for window_name, target_cols in windows_and_targets:
    print(f"\n--- Cross-validation for {window_name} window ---")
    
    for target_col in target_cols:
        X = df_cleaned.copy()

        if window_name == 'TE':
            X_window, y_window = prepare_X_y_te(X, target_col)
        else:
            X_window, y_window = prepare_X_y(X, target_col, window_name)

        groups = X_window['genome'] if data_filter.startswith("bygenome") else X_window.index
        
        print(f"\nTarget: {target_col}")
        print(f'Number of samples for this target: {X_window.shape[0]}')
        print(f'Number of features for this target: {X_window.shape[1]-1}')
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_window, groups=groups), 1):

            X_train, X_test = X_window.iloc[train_idx], X_window.iloc[test_idx]
            y_train, y_test = y_window.iloc[train_idx], y_window.iloc[test_idx] 

            if data_filter == "bygenome-seen":
                train_mask = X_train.index.isin(X_test.index)
                test_mask = X_test.index.isin(X_train.index)
                X_train, y_train = X_train[train_mask], y_train[train_mask]
                X_test, y_test = X_test[test_mask], y_test[test_mask]


            X_test_genome = X_test['genome']
            X_train = X_train.drop(columns=['genome'])
            X_test = X_test.drop(columns=['genome'])


            print(f'Fold: {fold}, train-test: {len(X_train)}/{len(X_test)}')

            if len(X_train) == 0 or len(X_test) < 2:
                print(f'  Skipping fold {fold}: not enough samples after filtering')
                continue

            
            model = create_model(X_train, 
                                mode=mode, 
                                modelname=modelname,
                                RANDOM_STATE_model=RANDOM_STATE_model)
            if modelname == 'RF':
                model.set_params(**best_params)

            # Training
            model.fit(X_train, y_train)

            # Prediction
            predictions = model.predict(X_test)
            
            # Store predictions
            fold_predictions.append(pd.DataFrame({
                                    'Window': window_name,
                                    'Target': target_col,
                                    'TE': X_test.index,
                                    'genome': X_test_genome,
                                    'Fold': fold,
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
            print('pearson score: ', pearson) 
            print('MAE: ', mae) 
            metrics = {
                    'Window': window_name,
                    'Target': target_col,
                    'Fold': fold,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'pearson': pearson  
                }

            fold_metrics.append(metrics)

        if not fold_metrics:
            print(f'WARNING: all folds skipped for {window_name} / {target_col}')
            continue

            
        # Save fold-level results
        fold_results[target_col] = fold_metrics
            
        # Average metrics across folds
        avg_metrics = pd.DataFrame(fold_metrics).drop(columns='Fold').mean(numeric_only=True).to_dict()
        evaluation_results[target_col] = avg_metrics
        
### Store results ###
results_df = pd.DataFrame(evaluation_results).T
results_df.to_csv(f'{OUT_PREFIX}_results.csv')
fold_results_df = pd.concat([pd.DataFrame(v) for v in fold_results.values()], ignore_index=True)
fold_results_df.to_csv(f'{OUT_PREFIX}_fold_results.csv', index=False)
all_predictions_df = pd.concat(fold_predictions, ignore_index=True)
all_predictions_df.to_csv(f'{OUT_PREFIX}_all_predictions.csv')