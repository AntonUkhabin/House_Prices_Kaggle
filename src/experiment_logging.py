from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from omegaconf import OmegaConf

from src.train_functions import get_boosting_training_info


def print_section(title: str) -> None:
    '''Print a formatted section title.'''

    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def print_experiment_info(config) -> None:
    '''Print experiment settings and target scale.'''

    print_section(f'Experiment: {config.general.experiment_name}')
    print(f'Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Random seed: {config.general.seed}')
    print(f'Fold seed: {config.training.fold_seed}')
    print(f'CV shuffle: {config.dataloader_params.shuffle}')
    print('Target transformation: np.log(SalePrice)')
    print('Primary metric: RMSE(log), lower is better')


def print_data_info(train_df, train_cv_df, holdout_df, test_df, config) -> None:
    '''Print dataset sizes and feature exclusions.'''

    print(f'Train: {train_df.shape}')
    print(f'Train/CV: {train_cv_df.shape}')
    print(f'Holdout: {holdout_df.shape}')
    print(f'Kaggle test: {test_df.shape}')
    print(f'Holdout fraction: {config.split.test_size}')
    if config.model.active == 'knn':
        print(f'Selected KNN features: {list(config.preprocessing.knn_features)}')
    else:
        print(f'Excluded columns: {list(config.preprocessing.drop_columns)}')


def print_model_info(config) -> None:
    '''Print the active model and its configured parameters.'''

    print_section('Model Information')
    print(f'Active model: {config.model.active}')
    for name, value in config.model.models[config.model.active].items():
        print(f'{name}: {value}')


def print_cv_start(config) -> None:
    '''Print the cross-validation heading and fold count.'''

    print_section('Cross Validation')
    print(f'Number of folds: {config.split.n_splits}')


def print_cv_summary(scores, number_of_models) -> None:
    '''Print mean CV error, its standard deviation and ensemble size.'''

    print(f'\nMean CV RMSE(log): {np.mean(scores):.5f}')
    print(f'CV STD: {np.std(scores):.5f}')
    print(f'Number of prediction models: {number_of_models}')


def print_model_diagnostics(fold_models, config, top_n=20):
    '''Print diagnostics specific to the active model.'''

    active_model = config.model.active

    if active_model in ('linear_regression', 'ridge', 'lasso', 'elastic_net'):
        return print_linear_coefficients(fold_models, top_n=top_n)

    if active_model == 'random_forest':
        return print_feature_importance(fold_models, top_n=top_n)

    if active_model == 'catboost':
        return print_catboost_diagnostics(fold_models, top_n=top_n)

    if active_model == 'xgboost':
        return print_xgboost_diagnostics(fold_models, config, top_n=top_n)

    if active_model == 'knn':
        return print_knn_diagnostics(fold_models, config)

    if active_model == 'dnn':
        return print_dnn_diagnostics(fold_models)

    raise ValueError(f'Unknown model for diagnostics: {active_model}')


def print_linear_coefficients(fold_models, top_n=20) -> pd.DataFrame:
    '''Print the largest linear coefficients averaged across folds.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    if top_n < 1:
        raise ValueError('top_n must be a positive integer.')

    fold_coefficients = []
    intercepts = []
    matrix_ranks = []
    feature_counts = []
    zero_coefficient_counts = []

    for fold, pipe in enumerate(fold_models, start=1):
        model = pipe.named_steps['model']
        preprocessor = pipe.named_steps['preprocessor']

        feature_names = preprocessor.get_feature_names_out()
        coefficients = np.asarray(model.coef_).reshape(-1)

        if len(feature_names) != len(coefficients):
            raise ValueError('Feature names and model coefficients have different lengths.')

        fold_coefficients.append(
            pd.Series(coefficients, index=feature_names, name=f'fold_{fold}')
        )

        intercepts.append(float(model.intercept_))
        # LinearRegression сохраняет ранг матрицы, а Ridge такого атрибута не предоставляет.
        if hasattr(model, 'rank_'):
            matrix_ranks.append(int(model.rank_))

        feature_counts.append(len(feature_names))
        zero_coefficient_counts.append(int(np.count_nonzero(coefficients == 0)))

    # Наборы One-Hot колонок могут различаться между folds из-за редких категорий.
    coefficients_by_fold = pd.concat(fold_coefficients, axis=1)

    coefficient_df = pd.DataFrame({
        'mean_coefficient': coefficients_by_fold.mean(axis=1),
        'mean_abs_coefficient': coefficients_by_fold.abs().mean(axis=1),
        'std_coefficient': coefficients_by_fold.std(axis=1, ddof=0),
        'fold_count': coefficients_by_fold.notna().sum(axis=1),
    })

    coefficient_df = (
        coefficient_df
        .sort_values('mean_abs_coefficient', ascending=False, kind='stable')
        .rename_axis('feature')
        .reset_index()
    )

    print_section('Linear Model Diagnostics')
    print(f'Mean intercept: {np.mean(intercepts):.5f}')
    print(f'Intercept STD: {np.std(intercepts):.5f}')
    print(f'Transformed features per fold: {min(feature_counts)}–{max(feature_counts)}')
    if any(zero_coefficient_counts):
        nonzero_coefficient_counts = np.asarray(feature_counts) - np.asarray(zero_coefficient_counts)
        print(f'Zero coefficients per fold: {min(zero_coefficient_counts)}–{max(zero_coefficient_counts)}')
        print(f'Non-zero coefficients per fold: {min(nonzero_coefficient_counts)}–{max(nonzero_coefficient_counts)}')
    if matrix_ranks:
        print(f'Design matrix rank per fold: {min(matrix_ranks)}–{max(matrix_ranks)}')
    print(f'Top {min(top_n, len(coefficient_df))} coefficients by mean absolute value:')
    print(coefficient_df.head(top_n).to_string(index=False, float_format=lambda value: f'{value:.5f}'))

    return coefficient_df


def print_feature_importance(fold_models, top_n=20) -> pd.DataFrame:
    '''Print top encoded features by mean impurity importance across folds.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    if top_n < 1:
        raise ValueError('top_n must be a positive integer.')

    fold_importances = []

    for pipe in fold_models:
        model = pipe.named_steps['model']
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()

        fold_importances.append(pd.Series(model.feature_importances_, index=feature_names))

    # Сопоставляем признаки по именам: набор one-hot колонок между фолдами может различаться.
    importance_by_fold = pd.concat(fold_importances, axis=1).fillna(0)

    importance_df = importance_by_fold.mean(axis=1).rename('importance').rename_axis('feature').reset_index()
    importance_df = importance_df.sort_values('importance', ascending=False, kind='stable').reset_index(drop=True)

    print_section('Feature Importance')
    print(f'Mean impurity-based importance across {len(fold_models)} folds (after encoding).')
    print(f'Top {min(top_n, len(importance_df))} features:')
    print(importance_df.head(top_n).to_string(index=False, float_format=lambda value: f'{value:.4f}'))

    return importance_df


def print_catboost_diagnostics(fold_models, top_n=20) -> pd.DataFrame:
    '''Print CatBoost training details and mean feature importance across folds.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    if top_n < 1:
        raise ValueError('top_n must be a positive integer.')

    fold_importances = []
    best_iterations = []
    tree_counts = []
    feature_counts = []
    categorical_counts = []

    for fold, pipe in enumerate(fold_models, start=1):
        model = pipe.named_steps['model']
        preprocessor = pipe.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()

        fold_importances.append(pd.Series(model.feature_importances_, index=feature_names, name=f'fold_{fold}'))
        best_iterations.append(model.get_best_iteration() + 1)
        tree_counts.append(model.tree_count_)
        feature_counts.append(len(feature_names))
        categorical_counts.append(len(preprocessor.named_steps['catboost_features'].categorical_features_))

    # Сопоставляем importance по названиям исходных признаков, затем усредняем по folds.
    importance_by_fold = pd.concat(fold_importances, axis=1)
    importance_df = importance_by_fold.mean(axis=1).rename('importance').rename_axis('feature').reset_index()
    importance_df = importance_df.sort_values('importance', ascending=False, kind='stable').reset_index(drop=True)

    print_section('CatBoost Diagnostics')
    print(f'Features per fold: {min(feature_counts)}–{max(feature_counts)}')
    print(f'Categorical features per fold: {min(categorical_counts)}–{max(categorical_counts)}')
    print(f'Best iterations: {best_iterations}')
    print(f'Trees retained: {tree_counts}')
    print(f'Mean CatBoost feature importance across {len(fold_models)} folds.')
    print(f'Top {min(top_n, len(importance_df))} features:')
    print(importance_df.head(top_n).to_string(index=False, float_format=lambda value: f'{value:.4f}'))

    return importance_df


def print_xgboost_diagnostics(fold_models, config, top_n=20) -> pd.DataFrame:
    '''Print XGBoost training details and normalized gain importance across folds.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    if top_n < 1:
        raise ValueError('top_n must be a positive integer.')

    fold_importances = []
    feature_counts = []
    best_iterations = []
    iterations_run = []
    iterations_used = []

    for fold, pipe in enumerate(fold_models, start=1):
        model = pipe.named_steps['model']
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        training_info = get_boosting_training_info(model, config)

        feature_counts.append(len(feature_names))
        best_iterations.append(training_info['best_iteration'])
        iterations_run.append(training_info['iterations_run'])
        iterations_used.append(training_info['iterations_used'])

        # Считаем gain только для раундов, используемых при predict после early stopping.
        booster = model.get_booster()[:training_info['iterations_used']]

        if booster.num_features() != len(feature_names):
            raise ValueError('Feature names and model input have different lengths.')

        gain_scores = booster.get_score(importance_type='gain')

        # При обучении на numpy XGBoost обозначает столбцы как f0, f1 и далее.
        booster_names = booster.feature_names
        if booster_names is None:
            booster_names = [f'f{index}' for index in range(len(feature_names))]

        importances = np.asarray([gain_scores.get(name, 0.0) for name in booster_names], dtype=float)

        # Нормируем каждый fold перед усреднением, чтобы привести важности к общей шкале.
        total_importance = importances.sum()
        if total_importance > 0:
            importances /= total_importance

        fold_importances.append(pd.Series(importances, index=feature_names, name=f'fold_{fold}'))

    # Редкие категории могут давать разные наборы OHE-столбцов между folds.
    importance_by_fold = pd.concat(fold_importances, axis=1).fillna(0)

    importance_df = importance_by_fold.mean(axis=1).rename('importance').rename_axis('feature').reset_index()
    importance_df = importance_df.sort_values('importance', ascending=False, kind='stable').reset_index(drop=True)

    print_section('XGBoost Diagnostics')
    print(f'Transformed features per fold: {min(feature_counts)}–{max(feature_counts)}')
    print(f'Best iterations: {best_iterations}')
    print(f'Iterations actually run: {iterations_run}')
    print(f'Iterations used for prediction: {iterations_used}')
    print(f'Mean normalized gain across {len(fold_models)} folds (after encoding, prediction rounds only).')
    print(f'Top {min(top_n, len(importance_df))} features:')
    print(importance_df.head(top_n).to_string(index=False, float_format=lambda value: f'{value:.4f}'))

    return importance_df


def print_knn_diagnostics(fold_models, config) -> pd.DataFrame:
    '''Print KNN feature space and fitted sample counts across folds.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    feature_names_by_fold = []
    fitted_sample_counts = []
    effective_metrics = []

    for pipe in fold_models:
        model = pipe.named_steps['model']
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out().tolist()

        feature_names_by_fold.append(feature_names)
        fitted_sample_counts.append(int(model.n_samples_fit_))
        effective_metrics.append(model.effective_metric_)

    reference_features = feature_names_by_fold[0]

    if any(feature_names != reference_features for feature_names in feature_names_by_fold[1:]):
        raise ValueError('KNN feature names differ between folds.')

    diagnostics_df = pd.DataFrame({'feature': reference_features})

    print_section('KNN Diagnostics')
    print(f'Features per fold: {len(reference_features)}')
    print(f'Fitted rows per fold: {min(fitted_sample_counts)}–{max(fitted_sample_counts)}')
    print(f'Neighbors: {config.model.models.knn.n_neighbors}')
    print(f'Weights: {config.model.models.knn.weights}')
    print(f'Configured metric: {config.model.models.knn.metric}, p={config.model.models.knn.p}')
    print(f'Effective metrics: {effective_metrics}')
    print('Features:')
    print(diagnostics_df.to_string(index=False))

    return diagnostics_df


def print_dnn_diagnostics(fold_models) -> pd.DataFrame:
    '''Print DNN architecture and fold-specific training diagnostics.'''

    if not fold_models:
        raise ValueError('No fitted DNN fold models provided.')

    diagnostics = []

    for fold_model in fold_models:
        model = fold_model.model
        dnn_preprocessor = fold_model.preprocessor.named_steps['preprocessor'].named_steps['dnn_features']
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

        diagnostics.append({
            'fold': fold_model.fold,
            'numerical_features': len(dnn_preprocessor.numerical_features_),
            'categorical_features': len(dnn_preprocessor.categorical_features_),
            'embedding_size': sum(model.embedding_dims),
            'trainable_parameters': trainable_parameters,
            'best_epoch': fold_model.best_epoch,
            'epochs_run': len(fold_model.history['validation_rmse']),
            'best_validation_rmse': fold_model.best_validation_rmse,
        })

    diagnostics_df = pd.DataFrame(diagnostics)

    reference_model = fold_models[0].model
    linear_layers = [layer for layer in reference_model.mlp if hasattr(layer, 'in_features')]
    mlp_dimensions = [linear_layers[0].in_features, *[layer.out_features for layer in linear_layers]]
    activations = [layer.__class__.__name__ for layer in reference_model.mlp if layer.__class__.__name__ not in ('Linear', 'Dropout')]
    dropout_rates = [layer.p for layer in reference_model.mlp if layer.__class__.__name__ == 'Dropout']

    print_section('DNN Diagnostics')
    print(f'MLP dimensions: {mlp_dimensions}')
    print(f'Activations: {activations}')
    print(f'Dropout rates: {dropout_rates}')
    print(f'Embedding dimensions: {reference_model.embedding_dims}')
    print(f'Best epochs: {diagnostics_df["best_epoch"].tolist()}')
    print(f'Epochs actually run: {diagnostics_df["epochs_run"].tolist()}')
    print(f'Trainable parameters per fold: {diagnostics_df["trainable_parameters"].tolist()}')
    print('Fold diagnostics:')
    print(diagnostics_df.to_string(index=False, float_format=lambda value: f'{value:.5f}'))

    return diagnostics_df


def save_boosting_training_history(fold_models, config) -> Path:
    '''Save boosting parameters and per-fold training histories to JSON.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    active_model = config.model.active

    history = {
        'experiment_name': config.general.experiment_name,
        'model': active_model,
        'target_transform': 'np.log',
        'metric': 'RMSE',
        'fold_seed': config.training.fold_seed,
        'parameters': OmegaConf.to_container(config.model.models[active_model], resolve=True),
        'folds': [],
    }

    for fold, pipe in enumerate(fold_models, start=1):
        model = pipe.named_steps['model']
        training_info = get_boosting_training_info(model, config)

        # Сохраняем всю историю, включая раунды ожидания после best iteration.
        history['folds'].append({
            'fold': fold,
            'best_iteration': training_info['best_iteration'],
            'iterations_run': training_info['iterations_run'],
            'iterations_used': training_info['iterations_used'],
            'train_rmse': [float(value) for value in training_info['train_rmse']],
            'validation_rmse': [float(value) for value in training_info['validation_rmse']],
        })

    output_dir = Path(config.paths.path_to_training_history)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{config.general.experiment_name}.json'

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=False)

    print(f'Training history saved: {output_path}')

    return output_path


def save_boosting_learning_curves(history_path) -> Path:
    '''Save per-fold learning curves alongside a boosting JSON history.'''

    history_path = Path(history_path)

    with history_path.open('r', encoding='utf-8') as file:
        history = json.load(file)

    folds = history['folds']

    if not folds:
        raise ValueError('Training history contains no folds.')

    figure, axes = plt.subplots(len(folds), 1, figsize=(12, 3.5 * len(folds)), squeeze=False)
    plot_path = history_path.with_suffix('.png')

    try:
        for axis, fold_history in zip(axes[:, 0], folds):
            train_rmse = np.asarray(fold_history['train_rmse'])
            validation_rmse = np.asarray(fold_history['validation_rmse'])
            iterations = np.arange(1, len(validation_rmse) + 1)

            # В JSON номер лучшей итерации сохранён с единицы, а индекс массива начинается с нуля.
            best_iteration = fold_history['best_iteration']
            best_rmse = validation_rmse[best_iteration - 1]

            axis.plot(iterations, train_rmse, label='Train RMSE')
            axis.plot(iterations, validation_rmse, label='Validation RMSE')
            axis.axvline(best_iteration, color='red', linestyle='--', alpha=0.7, label=f'Best iteration: {best_iteration}')

            axis.set_title(f'Fold {fold_history["fold"]} — RMSE(log) | Best validation: {best_rmse:.5f}')
            axis.set_xlabel('Iteration')
            axis.set_ylabel('RMSE(log)')
            axis.grid(alpha=0.3)
            axis.legend()

        model_name = {'catboost': 'CatBoost', 'xgboost': 'XGBoost'}.get(history['model'], history['model'])
        experiment_name = history['experiment_name']
        figure.suptitle(f'{model_name} training history — {experiment_name}', fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(plot_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(figure)

    print(f'Learning curves saved: {plot_path}')

    return plot_path


def print_submission_info(number_of_models) -> None:
    '''Print the ensemble prediction method and training scope.'''

    print_section('Kaggle Submission')
    print(f'Number of prediction models: {number_of_models}')
    print('Prediction method: average log predictions, then apply np.exp')
    print('Training data: Train/CV folds; holdout excluded')


def print_submission_summary(submission_df) -> None:
    '''Print the submission size and mean predicted price.'''

    print(f'Submission rows: {len(submission_df)}')
    print(f'Mean predicted price: ${submission_df["SalePrice"].mean():,.2f}')


def print_run_summary(config, elapsed_seconds) -> None:
    '''Print the completed run duration and log path.'''

    print_section('Run Completed')
    print(f'Total runtime: {elapsed_seconds:.1f} seconds')
    log_path = Path(config.paths.path_to_logs) / f'{config.general.experiment_name}.txt'
    print(f'Log saved: {log_path}')


def save_oof_predictions(train_cv_df, oof_predictions, fold_ids, config):
    '''Save out-of-fold predictions sorted by prediction error.'''

    oof_df = train_cv_df.copy()

    oof_df['fold'] = fold_ids + 1
    oof_df['actual_log'] = np.log(oof_df['SalePrice'])
    oof_df['prediction_log'] = oof_predictions
    oof_df['prediction_price'] = np.exp(oof_predictions)

    oof_df['error_log'] = oof_df['prediction_log'] - oof_df['actual_log']
    oof_df['abs_error_log'] = oof_df['error_log'].abs()
    oof_df['error_price'] = oof_df['prediction_price'] - oof_df['SalePrice']

    output_dir = Path(config.paths.path_to_oof)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = config.general.experiment_name
    errors_path = output_dir / f'{experiment_name}_errors.csv'

    # Сохраняем все OOF-наблюдения, начиная с объектов с наибольшей ошибкой.
    errors_df = oof_df.sort_values('abs_error_log', ascending=False, kind='stable').reset_index(drop=True)
    errors_df.to_csv(errors_path, index=False, sep=';', encoding='utf-8-sig')

    print(f'OOF errors saved: {errors_path}')

    return oof_df


def print_regression_metrics(name, metrics):
    '''Print regression metrics with explicit units.'''

    print(f'\n{name}:')
    print(f'RMSE(log): {metrics["rmse_log"]:.5f}')
    print(f'MAE($):    {metrics["mae_dollars"]:,.2f}')
    print(f'MSE($²):   {metrics["mse_dollars_squared"]:,.2f}')
    print(f'RMSE($):   {metrics["rmse_dollars"]:,.2f}')
    print(f'R²:        {metrics["r2"]:.5f}')
    print(f'MAPE:      {metrics["mape_pct"]:.2f}%')
    print(f'SMAPE:     {metrics["smape_pct"]:.2f}%')
    print(f'WAPE:      {metrics["wape_pct"]:.2f}%')


def save_holdout_predictions(holdout_df, predictions_log, config):
    '''Save holdout prices, predictions and prediction errors.'''

    predictions_log = np.asarray(predictions_log)

    predictions_df = holdout_df[['Id', 'SalePrice']].copy()
    predictions_df['actual_log'] = np.log(predictions_df['SalePrice'])
    predictions_df['prediction_log'] = predictions_log
    predictions_df['prediction_price'] = np.exp(predictions_log)

    predictions_df['error_log'] = predictions_df['prediction_log'] - predictions_df['actual_log']
    predictions_df['abs_error_log'] = predictions_df['error_log'].abs()
    predictions_df['error_price'] = predictions_df['prediction_price'] - predictions_df['SalePrice']

    output_dir = Path(config.paths.path_to_holdout)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{config.general.experiment_name}.csv'

    predictions_df = predictions_df.sort_values('abs_error_log', ascending=False, kind='stable').reset_index(drop=True)
    predictions_df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')

    print(f'Holdout predictions saved: {output_path}')

    return predictions_df


def save_submission(test_df, predictions_log, config):
    '''Convert log predictions to prices and save a Kaggle submission.'''

    predictions_price = np.exp(np.asarray(predictions_log))

    if predictions_price.shape != (len(test_df),):
        raise ValueError('Expected one prediction per test row.')

    if not np.isfinite(predictions_price).all() or (predictions_price <= 0).any():
        raise ValueError('Predicted prices must be finite and strictly positive.')

    submission_df = test_df[['Id']].copy()
    submission_df['SalePrice'] = predictions_price

    output_dir = Path(config.paths.path_to_submission)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{config.general.experiment_name}.csv'
    submission_df.to_csv(output_path, index=False)

    print(f'Submission saved: {output_path}')

    return submission_df


def save_dnn_training_history(fold_models, config) -> Path:
    '''Save DNN parameters, architectures and per-fold training histories to JSON.'''

    if not fold_models:
        raise ValueError('No fitted DNN fold models provided.')

    history = {
        'experiment_name': config.general.experiment_name,
        'model': 'dnn',
        'target_transform': 'np.log with fold-specific standardization',
        'metric': 'RMSE',
        'fold_seed': config.training.fold_seed,
        'parameters': OmegaConf.to_container(config.model.models.dnn, resolve=True),
        'folds': [],
    }

    for fold_model in fold_models:
        model = fold_model.model
        dnn_preprocessor = fold_model.preprocessor.named_steps['preprocessor'].named_steps['dnn_features']
        linear_layers = [layer for layer in model.mlp if hasattr(layer, 'in_features')]

        history['folds'].append({
            'fold': fold_model.fold,
            'best_epoch': fold_model.best_epoch,
            'epochs_run': len(fold_model.history['validation_rmse']),
            'best_validation_rmse': fold_model.best_validation_rmse,
            'target_mean': fold_model.target_mean,
            'target_std': fold_model.target_std,
            'numerical_features': len(dnn_preprocessor.numerical_features_),
            'categorical_features': len(dnn_preprocessor.categorical_features_),
            'embedding_dimensions': model.embedding_dims,
            'mlp_dimensions': [linear_layers[0].in_features, *[layer.out_features for layer in linear_layers]],
            'activations': [layer.__class__.__name__ for layer in model.mlp if layer.__class__.__name__ not in ('Linear', 'Dropout')],
            'dropout_rates': [layer.p for layer in model.mlp if layer.__class__.__name__ == 'Dropout'],
            'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'train_rmse': [float(value) for value in fold_model.history['train_rmse']],
            'validation_rmse': [float(value) for value in fold_model.history['validation_rmse']],
            'learning_rate': [float(value) for value in fold_model.history['learning_rate']],
        })

    output_dir = Path(config.paths.path_to_training_history)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{config.general.experiment_name}.json'

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=False)

    print(f'DNN training history saved: {output_path}')
    return output_path


def save_dnn_learning_curves(history_path) -> Path:
    '''Save per-fold DNN learning curves alongside their JSON history.'''

    history_path = Path(history_path)

    with history_path.open('r', encoding='utf-8') as file:
        history = json.load(file)

    folds = history['folds']

    if not folds:
        raise ValueError('DNN training history contains no folds.')

    figure, axes = plt.subplots(len(folds), 1, figsize=(12, 3.5 * len(folds)), squeeze=False)
    plot_path = history_path.with_suffix('.png')

    try:
        for axis, fold_history in zip(axes[:, 0], folds):
            train_rmse = np.asarray(fold_history['train_rmse'])
            validation_rmse = np.asarray(fold_history['validation_rmse'])
            epochs = np.arange(1, len(validation_rmse) + 1)
            best_epoch = fold_history['best_epoch']
            best_rmse = fold_history['best_validation_rmse']

            axis.plot(epochs, train_rmse, label='Train RMSE')
            axis.plot(epochs, validation_rmse, label='Validation RMSE')
            axis.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')

            axis.set_title(f'Fold {fold_history["fold"]} — RMSE(log) | Best validation: {best_rmse:.5f}')
            axis.set_xlabel('Epoch')
            axis.set_ylabel('RMSE(log)')
            axis.grid(alpha=0.3)
            axis.legend()

        figure.suptitle(f'DNN training history — {history["experiment_name"]}', fontsize=16)
        figure.tight_layout(rect=[0, 0, 1, 0.98])
        figure.savefig(plot_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(figure)

    print(f'DNN learning curves saved: {plot_path}')
    return plot_path