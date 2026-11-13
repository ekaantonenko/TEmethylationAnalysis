####################################
# --- Defining target variable --- #
####################################
regression_target_methylation_columns_TE = [
    'TE.Meth.CG', 'TE.Meth.CHG', 'TE.Meth.CHH',
    'TE.Meth.all'
]

regression_target_methylation_columns_0_100 = [
    #'Meth.CG.0_100', 'Meth.CHG.0_100', 'Meth.CHH.0_100',
    'Meth.all.0_100'
]

regression_target_methylation_columns_100_200 = [
    #'Meth.CG.100_200', 'Meth.CHG.100_200', 'Meth.CHH.100_200',
    'Meth.all.100_200'
]

regression_target_methylation_columns_200_300 = [
    #'Meth.CG.200_300', 'Meth.CHG.200_300', 'Meth.CHH.200_300',
    'Meth.all.200_300',
]


########################################
# --- Defining features per window --- #
########################################
features_per_window = {}
TE_features = [
        'Chr', 
        'TE.length', 'TE.length.ref', 'TE.length.%', 
        'TE.distance_to_pericentromere',
        'TE.superfamily', 
        'TE.inside_gene',
        'annotation'
        ]

gene_5p_features = [
        '5p.Gene1.length', '5p.Gene2.length',
        '5p.Gene1.oriented_to_te', '5p.Gene2.oriented_to_te',
        '5p.Gene1.distance', '5p.Gene2.distance'
        ]

gene_3p_features = [
        '3p.Gene1.length', '3p.Gene2.length',
        '3p.Gene1.oriented_to_te', '3p.Gene2.oriented_to_te',
        '3p.Gene1.distance', '3p.Gene2.distance'
        ]

gene_features = [
        'Gene1.length', 'Gene2.length',
        'Gene1.oriented_to_te', 'Gene2.oriented_to_te',
        'Gene1.distance', 'Gene2.distance'
        ]

### TE ###
features_per_window['TE'] = [
        *TE_features, 
        *gene_5p_features,
        *gene_3p_features,
        'TE.context.CG', 'TE.context.CHG', 'TE.context.CHH',
        '5p.context.CG.inside', '5p.context.CHG.inside', '5p.context.CHH.inside',
        '3p.context.CG.inside', '3p.context.CHG.inside', '3p.context.CHH.inside',
        'average.CG', 'average.CHG', 'average.CHH'
        ]



### First window 0-100bp ###
features_per_window['0_100'] = [
        *TE_features, 
        *gene_features,

        'TE.Meth.CG', 'TE.Meth.CHG', 'TE.Meth.CHH',
        'TE.context.CG', 'TE.context.CHG', 'TE.context.CHH', 

        'average.CG', 'average.CHG', 'average.CHH',   

        'Meth.CG.inside', 'Meth.CHG.inside', 'Meth.CHH.inside',
        'context.CG.inside', 'context.CHG.inside', 'context.CHH.inside',

        'context.CG.0_100', 'context.CHG.0_100', 'context.CHH.0_100'  
        ]

features_per_window['5p.0_100'] = [
        *TE_features,
        *gene_5p_features,
    
        'TE.Meth.CG', 'TE.Meth.CHG', 'TE.Meth.CHH', 
        'TE.context.CG', 'TE.context.CHG', 'TE.context.CHH', 

        'average.CG', 'average.CHG', 'average.CHH',     

        '5p.Meth.CG.inside', '5p.Meth.CHG.inside', '5p.Meth.CHH.inside',
        '5p.context.CG.inside', '5p.context.CHG.inside', '5p.context.CHH.inside',

        '5p.context.CG.0_100', '5p.context.CHG.0_100', '5p.context.CHH.0_100'
        ]
features_per_window['3p.0_100'] = [
        *TE_features,
        *gene_3p_features,
    
        'TE.Meth.CG', 'TE.Meth.CHG', 'TE.Meth.CHH', 
        'TE.context.CG', 'TE.context.CHG', 'TE.context.CHH', 

        'average.CG', 'average.CHG', 'average.CHH',     

        '3p.Meth.CG.inside', '3p.Meth.CHG.inside', '3p.Meth.CHH.inside',
        '3p.context.CG.inside', '3p.context.CHG.inside', '3p.context.CHH.inside',

        '3p.context.CG.0_100', '3p.context.CHG.0_100', '3p.context.CHH.0_100'
        ]

### Second window 100-200bp ###
features_per_window['100_200'] = [
        *features_per_window['0_100'],     
        'Meth.CG.0_100', 'Meth.CHG.0_100', 'Meth.CHH.0_100',
        #'context.CG.0_100', 'context.CHG.0_100', 'context.CHH.0_100',

        'context.CG.100_200', 'context.CHG.100_200', 'context.CHH.100_200'
        ]
features_per_window['5p.100_200'] = [
        *features_per_window['5p.0_100'],
        '5p.Meth.CG.0_100', '5p.Meth.CHG.0_100', '5p.Meth.CHH.0_100',
        #'5p.context.CG.0_100', '5p.context.CHG.0_100', '5p.context.CHH.0_100',

        '5p.context.CG.100_200', '5p.context.CHG.100_200', '5p.context.CHH.100_200'
        ]
features_per_window['3p.100_200'] = [
        *features_per_window['3p.0_100'],
        '3p.Meth.CG.0_100', '3p.Meth.CHG.0_100', '3p.Meth.CHH.0_100',
        #'3p.context.CG.0_100', '3p.context.CHG.0_100', '3p.context.CHH.0_100',

        '3p.context.CG.100_200', '3p.context.CHG.100_200', '3p.context.CHH.100_200'
        ]

### Third window 200-300bp ###
features_per_window['200_300'] = [
        *features_per_window['100_200'], 
        'Meth.CG.100_200', 'Meth.CHG.100_200', 'Meth.CHH.100_200',   
        #'context.CG.100_200', 'context.CHG.100_200', 'context.CHH.100_200', 

        'context.CG.200_300', 'context.CHG.200_300', 'context.CHH.200_300' 
        ]
features_per_window['5p.200_300'] = [
        *features_per_window['5p.100_200'], 
        '5p.Meth.CG.100_200', '5p.Meth.CHG.100_200', '5p.Meth.CHH.100_200', 
        #'5p.context.CG.100_200', '5p.context.CHG.100_200', '5p.context.CHH.100_200', 

        '5p.context.CG.200_300', '5p.context.CHG.200_300', '5p.context.CHH.200_300' 
        ]
features_per_window['3p.200_300'] = [
        *features_per_window['3p.100_200'], 
        '3p.Meth.CG.100_200', '3p.Meth.CHG.100_200', '3p.Meth.CHH.100_200',   
        #'3p.context.CG.100_200', '3p.context.CHG.100_200', '3p.context.CHH.100_200',

        '3p.context.CG.200_300', '3p.context.CHG.200_300', '3p.context.CHH.200_300'   
        ]


#############################################
# --- Dictionary of windows and targets --- #
#############################################
regression_flanks_windows_and_targets = [
    ('0_100', regression_target_methylation_columns_0_100),
    ('100_200', regression_target_methylation_columns_100_200),
    ('200_300', regression_target_methylation_columns_200_300)
]

regression_TE_windows_and_targets = [
    ('TE', regression_target_methylation_columns_TE)
]

