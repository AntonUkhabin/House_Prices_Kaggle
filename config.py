from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 0xC0FFEE,
        'experiment_name': '2_linear_regression_baseline',
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
    }, 
    'preprocessing': {
        'drop_columns': [],
    },
    'model': {
        'active': 'linear_regression',

        'models': {

            'linear_regression': {
                'fit_intercept': True,
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