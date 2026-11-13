import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression




def create_model(X,
                 mode,
                 modelname,
                 RANDOM_STATE_model=42):
    """
    Create a model for the given mode and modelname.

    For modelname='RF':
        Returns an sklearn Pipeline (OrdinalEncoder + StandardScaler + RandomForest).
        'TE.sequence' must be dropped from X before calling (done in crossvalidation script).

    """
    if modelname == 'RF':
        return _create_rf_model(X, mode, RANDOM_STATE_model)
    elif modelname == 'linear':
        return _create_linear_model(X, mode)
    elif modelname == 'mean-genome':
        return _create_meangenome_model(X, mode)
    else:
        raise ValueError(f"Unknown modelname: {modelname}")

########################################################################################
# Random Forest model
########################################################################################


def _create_rf_model(X, mode, RANDOM_STATE_model=42):
    """Original RF pipeline (unchanged logic)."""
    numeric_cols     = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
            #('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ],
        remainder='passthrough'
    )

    if mode.startswith('regression'):
        predictor = ('regressor', RandomForestRegressor(
            random_state=RANDOM_STATE_model,
            n_jobs=-1
        ))
    elif mode.startswith('binary'):
        predictor = ('classifier', RandomForestClassifier(
            random_state=RANDOM_STATE_model,
            n_jobs=-1
        ))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Pipeline([('preprocessor', preprocessor), predictor])

########################################################################################
# Baseline: Linear regression ("linear")
########################################################################################


def _create_linear_model(X, mode):
    """ Only for continuous model """
    numeric_cols     = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            #('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ],
        remainder='passthrough'
    )

    if mode.startswith('regression'):
        predictor = ('regressor', LinearRegression())
    elif mode.startswith('binary'):
        raise ValueError(f"Binary mode not implemented")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return Pipeline([('preprocessor', preprocessor), predictor])



########################################################################################
# Baseline: per-TIS average methylation ("mean-genome")
########################################################################################


class TISMeanRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline for genome-grouped cross-validation.

    For each TE in the test set, predicts the average methylation of the same
    TIS in the training genomes (i.e. in the other genomes). The TIS identifier
    is read from the DataFrame index, as in the CV script.

    TISs not seen in training (e.g. insertions present only in test genomes)
    fall back to the average of `fallback_col` (e.g. superfamily) if given,
    and otherwise to the global training average.

    Parameters
    ----------
    statistic : {'mean', 'median'}
        Aggregation per TIS. 'mean' minimizes MSE, 'median' minimizes MAE.
    fallback_col : str or None
        Column used for unseen TISs, e.g. 'TE.superfamily'.
    extra_key_cols : list of str or None
        Extra columns combined with the TIS index to form the key, e.g. a
        flank-direction column when each TE appears twice (upstream/downstream).
    """

    def __init__(self, statistic='mean', fallback_col=None, extra_key_cols=None):
        self.statistic = statistic
        self.fallback_col = fallback_col
        self.extra_key_cols = extra_key_cols

    def _keys(self, X):
        keys = pd.DataFrame({'_tis': np.asarray(X.index)})
        for col in (self.extra_key_cols or []):
            keys[col] = np.asarray(X[col])
        return keys

    def fit(self, X, y):
        if self.statistic not in ('mean', 'median'):
            raise ValueError("statistic must be 'mean' or 'median'")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TISMeanRegressor needs a DataFrame indexed by TIS ID")

        y = np.asarray(y, dtype=float)
        keys = self._keys(X)
        self.key_cols_ = list(keys.columns)
        keys['_y'] = y

        # Per-TIS value learned from the training genomes
        self.tis_values_ = (keys.groupby(self.key_cols_)['_y']
                                .agg(self.statistic)
                                .rename('_pred')
                                .reset_index())

        # Fallbacks for TISs absent from training
        self.global_value_ = float(pd.Series(y).agg(self.statistic))
        if self.fallback_col is not None:
            self.fallback_values_ = (pd.Series(y, index=np.asarray(X[self.fallback_col]))
                                       .groupby(level=0)
                                       .agg(self.statistic))

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        keys = self._keys(X)
        # Left merge preserves the row order of X
        pred = (keys.merge(self.tis_values_, on=self.key_cols_, how='left')['_pred']
                    .to_numpy(dtype=float, copy=True))

        unseen = np.isnan(pred)
        if self.fallback_col is not None and unseen.any():
            fb_keys = pd.Series(np.asarray(X[self.fallback_col])[unseen])
            pred[unseen] = fb_keys.map(self.fallback_values_).to_numpy(dtype=float)
        pred = np.where(np.isnan(pred), self.global_value_, pred)

        # Diagnostics: how many test TEs had no training instance of their TIS
        self.last_n_unseen_ = int(unseen.sum())
        self.last_frac_unseen_ = self.last_n_unseen_ / len(pred) if len(pred) else 0.0
        return pred


def _create_meangenome_model(X, mode):
    """ Only for continuous model
        Intended for 'bygenome'-grouped predictions:
        predicts the average methylation per TIS, if seen in other genomes.
        Unseen TISs fall back to the superfamily average (if the column exists),
        otherwise to the global average.
    """
    if not mode.startswith('regression'):
        raise ValueError("The 'mean-genome' baseline is implemented for regression modes only")

    fallback_col = 'TE.superfamily' if 'TE.superfamily' in X.columns else None
    return TISMeanRegressor(statistic='mean', fallback_col=fallback_col)


########################################################################################
# Gridsearch model
########################################################################################

def create_model_gridsearch(X, mode, RANDOM_STATE_RForest=42):
    """Grid-search variant — RF only (unchanged)."""
    return _create_rf_model(X, mode, RANDOM_STATE_RForest)