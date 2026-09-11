import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline           import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from catboost import CatBoostRegressor

from src.preprocessing          import build_preprocessor
from src.feature_engineering import SelectedLog1pTransformer


def build_model(config, categorical_features=None):
    '''Build estimator from config.'''

    active_model = config.model.active
    model_params = dict(config.model.models[active_model])

    if active_model == 'linear_regression':
        return LinearRegression(**model_params)

    if active_model == 'ridge':
        return Ridge(**model_params)

    if active_model == 'lasso':
        return Lasso(**model_params)

    if active_model == 'elastic_net':
        return ElasticNet(**model_params)
    
    if active_model == 'random_forest':
        return RandomForestRegressor(**model_params)

    if active_model == 'catboost':
        if categorical_features is None:
            raise ValueError('Categorical features are required for CatBoost.')

        return CatBoostRegressor(cat_features=list(categorical_features), **model_params)

    raise ValueError(f'Unknown model name: {active_model}')


def build_feature_pipeline(config) -> Pipeline:
    '''Build model-specific feature transformations without an estimator.'''

    steps = []

    # Сохраняем прежний log1p для существующих моделей; CatBoost получает признаки в исходной шкале.
    if config.model.active != 'catboost':
        steps.append(('selected_log1p', SelectedLog1pTransformer()))

    steps.append(('preprocessor', build_preprocessor(config)))

    return Pipeline(steps)


def build_pipeline(config) -> Pipeline:
    '''Build the full preprocessing and model pipeline.'''

    feature_pipeline = build_feature_pipeline(config)
    model = build_model(config)

    return Pipeline(feature_pipeline.steps + [('model', model)])


def fit_model_with_early_stopping(model, features_train, labels_train, features_val, labels_val, config):
    '''Fit a boosting estimator using a validation set for early stopping.'''

    if config.model.active == 'catboost':
        model.fit(features_train, labels_train, eval_set=(features_val, labels_val))
        return model

    raise ValueError(f'Early stopping is not supported for model: {config.model.active}')


def fit_pipeline_with_early_stopping(features_train, labels_train, features_val, labels_val, config) -> Pipeline:
    '''Fit fold preprocessing and a boosting estimator, returning a fitted pipeline.'''

    feature_pipeline = build_feature_pipeline(config)

    # Все обучаемые преобразования выполняют fit только на train-фолде.
    features_train_transformed = feature_pipeline.fit_transform(features_train, labels_train)
    features_val_transformed = feature_pipeline.transform(features_val)

    categorical_features = None

    if config.model.active == 'catboost':
        catboost_preprocessor = feature_pipeline.named_steps['preprocessor'].named_steps['catboost_features']
        categorical_features = catboost_preprocessor.categorical_features_

    model = build_model(config, categorical_features=categorical_features)

    model = fit_model_with_early_stopping(
        model=model,
        features_train=features_train_transformed,
        labels_train=labels_train,
        features_val=features_val_transformed,
        labels_val=labels_val,
        config=config,
    )

    # Собираем уже обученные объекты: повторный fit здесь не нужен.
    return Pipeline(feature_pipeline.steps + [('model', model)])


def cross_validate_standard(train_cv_df, target_col, config):
    '''Run cross-validation on development data, excluding holdout.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = np.log(train_cv_df[target_col])

    shuffle = config.dataloader_params.shuffle
    fold_seed = config.training.fold_seed if shuffle else None

    kfold = KFold(n_splits=config.split.n_splits, shuffle=shuffle, random_state=fold_seed)

    scores = []
    fold_models = []
    oof_predictions = np.full(len(train_cv_df), np.nan)
    fold_ids = np.full(len(train_cv_df), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]
        labels_train = labels.iloc[train_idx]
        labels_val = labels.iloc[val_idx]

        fold_pipe = build_pipeline(config)
        fold_pipe.fit(features_train, labels_train)

        predictions_log = fold_pipe.predict(features_val)
        score = root_mean_squared_error(labels_val, predictions_log)

        oof_predictions[val_idx] = predictions_log
        fold_ids[val_idx] = fold

        scores.append(float(score))
        fold_models.append(fold_pipe)

        print(f'Fold {fold + 1}: RMSE(log) = {score:.5f}')

    return scores, fold_models, oof_predictions, fold_ids


def cross_validate_model_with_early_stopping(train_cv_df, target_col, config):
    '''Run cross-validation with fold-specific early stopping and OOF predictions.'''

    features = train_cv_df.drop(columns=[target_col])
    labels = np.log(train_cv_df[target_col])

    shuffle = config.dataloader_params.shuffle
    fold_seed = config.training.fold_seed if shuffle else None
    kfold = KFold(n_splits=config.split.n_splits, shuffle=shuffle, random_state=fold_seed)

    scores = []
    fold_models = []
    oof_predictions = np.full(len(train_cv_df), np.nan)
    fold_ids = np.full(len(train_cv_df), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]
        labels_train = labels.iloc[train_idx]
        labels_val = labels.iloc[val_idx]

        fold_pipe = fit_pipeline_with_early_stopping(
            features_train=features_train,
            labels_train=labels_train,
            features_val=features_val,
            labels_val=labels_val,
            config=config,
        )

        predictions_log = fold_pipe.predict(features_val)
        score = root_mean_squared_error(labels_val, predictions_log)

        # Записываем OOF по позициям исходных строк, независимо от индекса DataFrame.
        oof_predictions[val_idx] = predictions_log
        fold_ids[val_idx] = fold

        scores.append(float(score))
        fold_models.append(fold_pipe)

        model = fold_pipe.named_steps['model']

        # CatBoost возвращает индекс лучшей итерации с нуля; в логе показываем номер с единицы.
        best_iteration = model.get_best_iteration() + 1
        iterations_run = len(model.get_evals_result()['validation']['RMSE'])

        print(f'Fold {fold + 1}: RMSE(log) = {score:.5f}, best iteration = {best_iteration}, iterations run = {iterations_run}, trees retained = {model.tree_count_}')

    return scores, fold_models, oof_predictions, fold_ids


def predict_with_pipeline_ensemble(features, fold_models):
    '''Predict log prices by averaging fitted fold models.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    fold_predictions = [pipe.predict(features) for pipe in fold_models]

    return np.mean(fold_predictions, axis=0)


def calculate_regression_metrics(labels_true, predictions_log):
    '''Calculate log-space RMSE and price-space regression metrics.'''

    labels_true = np.asarray(labels_true, dtype=float)
    predictions_log = np.asarray(predictions_log, dtype=float)

    if labels_true.ndim != 1 or labels_true.shape != predictions_log.shape:
        raise ValueError('Labels and predictions must be matching one-dimensional arrays.')

    if not np.isfinite(labels_true).all() or (labels_true <= 0).any():
        raise ValueError('Actual prices must be finite and strictly positive.')

    predictions_price = np.exp(predictions_log)
    absolute_errors = np.abs(labels_true - predictions_price)

    return {
        'rmse_log': root_mean_squared_error(np.log(labels_true), predictions_log),
        'mae_dollars': mean_absolute_error(labels_true, predictions_price),
        'mse_dollars_squared': mean_squared_error(labels_true, predictions_price),
        'rmse_dollars': root_mean_squared_error(labels_true, predictions_price),
        'r2': r2_score(labels_true, predictions_price),
        'mape_pct': mean_absolute_percentage_error(labels_true, predictions_price) * 100,
        'smape_pct': np.mean(2 * absolute_errors / (np.abs(labels_true) + np.abs(predictions_price))) * 100,
        'wape_pct': absolute_errors.sum() / np.abs(labels_true).sum() * 100,
    }