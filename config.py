from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '65_blend_lasso_catboost_dnn',
    },
    'paths': {
        'path_to_csv':                  './data/train.csv',
        'path_to_kaggle_test':          './data/test.csv',
        
        'path_to_logs':                 './outputs/logs',
        'path_to_oof':                  './outputs/oof',
        'path_to_holdout':              'outputs/holdout',
        'path_to_submission':           'outputs/submissions',
        'path_to_training_history':     'outputs/training_history',
        'path_to_checkpoints':          'outputs/checkpoints',
        'path_to_shap':                 './outputs/shap',
        'path_to_permutation_importance': './outputs/permutation_importance',
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
        'training_log_interval': 10,  # Print training metrics every N epochs. 10 - print every 10 epochs, 1 - print every epoch.
    },
    'shap': {
        # OOF SHAP поддерживается только для одиночного запуска CatBoost.
        'enabled': False,
        'max_display': 20,
    },
    'permutation_importance': {
        # OOF Permutation Importance поддерживается для одиночных моделей, но не для blend.
        'enabled': False,
        'n_repeats': 5,
    },
    'split': {
        'n_splits': 5,
        'test_size': 0.2,

        # Включает удаление двух аномальных продаж, найденных во время EDA.
        'remove_eda_outliers': True,
    }, 
    'preprocessing': {
        'drop_columns': ['Utilities', 'Condition2', 'BsmtUnfSF', 'BldgType', 'HouseStyle', 'Exterior2nd', 'GarageQual',
            'Street', 'Alley', 'RoofMatl', 'Heating', 'LowQualFinSF', 'PoolQC', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces'],

        # DNN использует все исходные признаки; сюда добавляются только исключённые DNN-specific features.
        'dnn_drop_columns': [],
    },
    'blending': {
        'weights': {
            'lasso': 0.400,
            'catboost': 0.275,
            'dnn': 0.325,
        },
    },
    'model': {
        'active': 'blend', # 'blend' for Ensemble

        'models': {

            'dnn': {
                'epochs': 500,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'early_stopping_rounds': 30,
                'min_delta': 0.0001,
                'num_workers': 0,
            },

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
            
            'xgboost': {
                'n_estimators': 10000,
                'learning_rate': 0.053,
                'max_depth': 2,
                'min_child_weight': 3,
                'subsample': 0.82,
                'colsample_bytree': 0.686,
                'reg_alpha': 0.0,
                'reg_lambda': 4.06,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'early_stopping_rounds': 200,
                'tree_method': 'hist',
                'importance_type': 'gain',
                'random_state': '${general.seed}',
                'n_jobs': 15,
                'verbosity': 1,
            },
        },
    },
}

config = OmegaConf.create(config)