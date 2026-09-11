from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '43_catboost_lr_01_optuna_tuned',
    },
    'paths': {
        'path_to_csv':                  './data/train.csv',
        'path_to_kaggle_test':          './data/test.csv',
        
        'path_to_logs':                 './outputs/logs',
        'path_to_oof':                  './outputs/oof',
        'path_to_holdout':              'outputs/holdout',
        'path_to_submission':           'outputs/submissions',
        'path_to_training_history':     'outputs/training_history',
        'path_to_shap':                 './outputs/shap',
    },
    'training': {
        'fold_seed': 0xC0FFEE,
    },
    'evaluation': {
        # Holdout остаётся исключённым из обучения независимо от этого переключателя.
        'evaluate_holdout': True,
    },
    'logging': {
        'save_training_history': True,
    },
    'shap': {
        'enabled': True,
        'max_display': 20,
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
        'active': 'catboost',

        'models': {

            'linear_regression': {
                'fit_intercept': True,
            },

            'ridge': {
                'alpha': 15.0,
                'fit_intercept': True,
            },

            'lasso': {
                'alpha': 0.000225,
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
                'n_estimators': 900,
                'criterion': 'squared_error',
                'max_depth': 12,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 0.4,
                'bootstrap': False,
                'random_state': 0xC0FFEE,
                'n_jobs': 15,
            },

            'catboost': {
                'iterations': 10000,
                'learning_rate': 0.1,
                'depth': 4,
                'l2_leaf_reg': 2.21,
                'random_strength': 1.99,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'early_stopping_rounds': 200,
                'use_best_model': True,
                'nan_mode': 'Min',
                'random_seed': '${general.seed}',
                'task_type': 'CPU',
                'thread_count': 15,
                'verbose': False,
                'allow_writing_files': False,
            }, 
        },
    },
}

config = OmegaConf.create(config)