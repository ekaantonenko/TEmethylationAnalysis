import pandas as pd 

from _features_and_targets import features_per_window

def applyFilterContext(df, window, direction=None):
    if window == 'TE':
        return df[
        (df[f'TE.context.CG'] >= 2/100) &
        (df[f'TE.context.CHG'] >= 2/100) &
        (df[f'TE.context.CHH'] >= 4/100) 
        ]
    else:
        return df[
        (df[f'{direction}.context.CG.{window}'] >= 2/100) &
        (df[f'{direction}.context.CHG.{window}'] >= 2/100) &
        (df[f'{direction}.context.CHH.{window}'] >= 4/100) 
        ]

def applyAllContextFilters(X, target_col, window_name):
    ### Filter out insufficient contexts
    direction = target_col.split('.')[0]
    X = applyFilterContext(X, 'inside', direction)
    if window_name == '100_200':
        X = applyFilterContext(X, '0_100', direction)
    if window_name == '200_300':
        X = applyFilterContext(X, '0_100', direction)
        X = applyFilterContext(X, '100_200', direction)

    # Filter out insufficient contexts IF predicting cg/chg/chh 
    if 'all' in target_col:
        X_window = X.copy()
    else:
        X_window = applyFilterContext(X, window_name, direction)
    
    return X_window


def prepare_X_y(X, target_col, window_name, filter=True):
    if filter == True:
        X_window_5p = applyAllContextFilters(X, f'5p.{target_col}', window_name)
        X_window_3p = applyAllContextFilters(X, f'3p.{target_col}', window_name)    

    y_window_5p = X_window_5p[f'5p.{target_col}']
    y_window_3p = X_window_3p[f'3p.{target_col}']
    X_window_5p = X_window_5p[[*features_per_window[f'5p.{window_name}'], 'TE.ID_tair10', 'genome']]
    X_window_3p = X_window_3p[[*features_per_window[f'3p.{window_name}'], 'TE.ID_tair10', 'genome']]    

    cols_no_direction_5p = [col.lstrip('5p.') for col in X_window_5p.columns]
    cols_no_direction_3p = [col.lstrip('3p.') for col in X_window_3p.columns]

    assert(cols_no_direction_5p == cols_no_direction_3p)
    cols_no_direction = cols_no_direction_5p 
    X_window_5p.columns = cols_no_direction
    X_window_3p.columns = cols_no_direction
    y_window_5p.columns = [target_col]
    y_window_3p.columns = [target_col]

    X_window = pd.concat([X_window_5p, X_window_3p])  
    y_window = pd.concat([y_window_5p, y_window_3p])

    X_window = X_window.set_index('TE.ID_tair10')
    y_window.index = X_window.index

    return (X_window, y_window)

def prepare_X_y_te(X, target_col, filter=True):

    #X_window = applyFilterContext(X, window='TE')
    
    #X_window = applyFilterContext(X_window, window='inside', direction='5p')
    #X_window = applyFilterContext(X_window, window='inside', direction='3p')
    
    
    X_window = X[[*features_per_window['TE'], 'TE.ID_tair10', 'genome', target_col]]

    if filter==True:
        X_window = X_window.dropna()
    print(len(X_window))

    y_window = X_window[target_col]
    X_window = X_window.drop(columns=[target_col])

    X_window = X_window.set_index('TE.ID_tair10')
    y_window.index = X_window.index
    return (X_window, y_window)