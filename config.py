from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '29_elastic_net_return_garage_condition',
    },
    'paths': {
        'path_to_csv':                  './data/train.csv',
        'path_to_kaggle_test':          './data/test.csv',
        'path_to_logs':                 './outputs/logs',
        'path_to_oof':                  './outputs/oof',
        'path_to_holdout':              'outputs/holdout',
        'path_to_submission':           'outputs/submissions',
    },
    'training': {
        'early_stopping_rounds': 500, # Patience for gradient boosting models.
        'fold_seed': 0xC0FFEE,
    },
    'dataloader_params': {
        'shuffle': True,
    },
    'split': {
        'n_splits': 5,
        'test_size': 0.2,

        # Включает удаление двух аномальных продаж, найденных во время EDA.
        'remove_eda_outliers': True,
    }, 
    'preprocessing': {
        'drop_columns': ['Utilities', 'Condition2', 'BsmtUnfSF', 'BldgType', 'HouseStyle', 'Exterior2nd', 'GarageQual',
            'Street', 'Alley', 'RoofMatl', 'Heating', 'LowQualFinSF', 'PoolQC', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'MasVnrArea',
            'Fireplaces'],
    },
    'model': {
        'active': 'elastic_net',

        'models': {

            'linear_regression': {
                'fit_intercept': True,
            },

            'ridge': {
                'alpha': 15.0,
                'fit_intercept': True,
            },

            'lasso': {
                'alpha': 0.00025,
                'fit_intercept': True,
                'max_iter': 50_000,
                'tol': 1e-4,
                'selection': 'cyclic',
            },

            'elastic_net': {
                'alpha': 0.0004,
                'l1_ratio': 0.9,
                'fit_intercept': True,
                'max_iter': 50_000,
                'tol': 1e-4,
                'selection': 'cyclic',
            },

            'random_forest': {
                'n_estimators': 300,
                'criterion': 'squared_error',
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 1.0,
                'bootstrap': True,
                'oob_score': False,
                'random_state': 0xC0FFEE,
                'n_jobs': -1,
            },
        },
    },
}

config = OmegaConf.create(config)