from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from omegaconf import OmegaConf


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


def save_catboost_training_history(fold_models, config) -> Path:
    '''Save CatBoost parameters and per-fold training histories to JSON.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    history = {
        'experiment_name': config.general.experiment_name,
        'model': config.model.active,
        'target_transform': 'np.log',
        'metric': 'RMSE',
        'fold_seed': config.training.fold_seed,
        'parameters': OmegaConf.to_container(config.model.models.catboost, resolve=True),
        'folds': [],
    }

    for fold, pipe in enumerate(fold_models, start=1):
        model = pipe.named_steps['model']
        evals_result = model.get_evals_result()

        # Сохраняем всю историю, включая patience после лучшей итерации, а не только сохранённые деревья.
        history['folds'].append({
            'fold': fold,
            'best_iteration': int(model.get_best_iteration() + 1),
            'trees_retained': int(model.tree_count_),
            'train_rmse': [float(value) for value in evals_result['learn']['RMSE']],
            'validation_rmse': [float(value) for value in evals_result['validation']['RMSE']],
        })

    output_dir = Path(config.paths.path_to_training_history)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{config.general.experiment_name}.json'

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=False)

    print(f'Training history saved: {output_path}')

    return output_path


def save_catboost_learning_curves(history_path) -> Path:
    '''Save per-fold learning curves alongside a CatBoost JSON history.'''

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

        figure.suptitle(f'CatBoost training history — {history["experiment_name"]}', fontsize=14)
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
