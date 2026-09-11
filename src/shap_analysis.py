import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool


def calculate_model_shap(model, features):
    '''Return feature contributions and base values for a supported regression model.'''

    if isinstance(model, CatBoostRegressor):
        pool = Pool(features, cat_features=model.get_cat_feature_indices())
        thread_count = model.get_params().get('thread_count', -1)
        shap_values = model.get_feature_importance(pool, type='ShapValues', thread_count=thread_count)

        # Последний столбец CatBoost содержит base value, остальные — вклады признаков.
        return shap_values[:, :-1], shap_values[:, -1]

    raise ValueError(f'SHAP is not supported for model: {type(model).__name__}')


def calculate_oof_shap(train_cv_df, target_col, fold_models, fold_ids):
    '''Calculate validation-fold SHAP values aligned with the original development rows.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    features = train_cv_df.drop(columns=[target_col])
    fold_ids = np.asarray(fold_ids)

    if fold_ids.shape != (len(features),):
        raise ValueError('Expected one fold ID per development row.')

    if not np.array_equal(np.unique(fold_ids), np.arange(len(fold_models))):
        raise ValueError('Fold IDs must match the supplied fold models.')

    shap_parts = []
    feature_parts = []
    base_values = np.full(len(features), np.nan)

    for fold, pipe in enumerate(fold_models):
        val_idx = np.flatnonzero(fold_ids == fold)

        # Используем только validation-строки и уже обученный preprocessing соответствующего фолда.
        features_val = pipe[:-1].transform(features.iloc[val_idx])

        if not isinstance(features_val, pd.DataFrame):
            raise TypeError('OOF SHAP currently requires preprocessing with DataFrame output.')

        model = pipe.named_steps['model']
        shap_values, fold_base_values = calculate_model_shap(model, features_val)

        # Контролируем, что SHAP объясняет именно прогноз модели в её выходной шкале.
        reconstructed_predictions = fold_base_values + shap_values.sum(axis=1)
        np.testing.assert_allclose(reconstructed_predictions, model.predict(features_val), rtol=1e-6, atol=1e-8)

        shap_parts.append(pd.DataFrame(shap_values, index=val_idx, columns=features_val.columns))

        features_val = features_val.copy()
        features_val.index = val_idx
        feature_parts.append(features_val)

        base_values[val_idx] = fold_base_values

        print(f'Fold {fold + 1}: SHAP calculated for {len(val_idx)} validation rows')

    # Общие графики требуют одинакового набора признаков во всех фолдах.
    feature_names = shap_parts[0].columns

    if any(not part.columns.equals(feature_names) for part in shap_parts):
        raise ValueError('Feature columns must match across folds for OOF SHAP.')

    shap_df = pd.concat(shap_parts).sort_index()
    feature_df = pd.concat(feature_parts).sort_index()

    # Возвращаем исходный порядок строк; Id сохраняем отдельно для сопоставления с OOF errors.
    return {
        'ids': train_cv_df['Id'].to_numpy(copy=True),
        'fold_ids': fold_ids.copy(),
        'features': feature_df,
        'shap_values': shap_df,
        'base_values': base_values,
    }


def save_oof_shap(shap_result, config):
    '''Save OOF SHAP contributions, feature values and global importance.'''

    output_dir = Path(config.paths.path_to_shap)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = config.general.experiment_name
    shap_df = shap_result['shap_values']

    values_path = output_dir / f'{experiment_name}_shap.npz'
    features_path = output_dir / f'{experiment_name}_features.csv'
    importance_path = output_dir / f'{experiment_name}_importance.csv'

    # NPZ сохраняет числовые массивы без потери точности; все строки имеют единый порядок.
    np.savez_compressed(
        values_path,
        shap_values=shap_df.to_numpy(),
        base_values=shap_result['base_values'],
        ids=shap_result['ids'],
        fold_ids=shap_result['fold_ids'],
        feature_names=np.asarray(shap_df.columns, dtype=str),
    )

    feature_df = shap_result['features'].copy()
    feature_df.insert(0, 'Id', shap_result['ids'])
    feature_df.to_csv(features_path, index=False, sep=';', encoding='utf-8-sig')

    # Усредняем модули вкладов: положительные и отрицательные SHAP не компенсируют друг друга.
    importance_df = shap_df.abs().mean(axis=0).rename('mean_abs_shap').rename_axis('feature').reset_index()
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False, kind='stable').reset_index(drop=True)
    importance_df.to_csv(importance_path, index=False, sep=';', encoding='utf-8-sig')

    print(f'OOF SHAP saved: {values_path}')
    print(f'SHAP feature values saved: {features_path}')
    print(f'SHAP importance saved: {importance_path}')

    return importance_df


def save_shap_importance_plot(importance_df, config) -> Path:
    '''Save a horizontal bar chart of mean absolute OOF SHAP contributions.'''

    max_display = config.shap.max_display

    if max_display < 1:
        raise ValueError('max_display must be positive.')

    if importance_df.empty:
        raise ValueError('SHAP importance is empty.')

    output_dir = Path(config.paths.path_to_shap)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f'{config.general.experiment_name}_importance.png'

    # Переворачиваем top-N, чтобы самый важный признак оказался наверху горизонтального графика.
    displayed = importance_df.head(max_display).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(displayed) + 1.5)))

    try:
        axis.barh(displayed['feature'], displayed['mean_abs_shap'], color='#4285B4')
        axis.set_xlabel('Mean |SHAP value| — log(SalePrice)')
        axis.set_title(f'OOF SHAP importance — {config.general.experiment_name}')
        axis.set_axisbelow(True)
        axis.grid(axis='x', alpha=0.3)

        figure.tight_layout()
        figure.savefig(plot_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(figure)

    print(f'SHAP importance plot saved: {plot_path}')

    return plot_path


def save_shap_beeswarm_plot(shap_result, config) -> Path:
    '''Save an OOF SHAP beeswarm plot with numerical feature colors.'''

    import shap

    max_display = config.shap.max_display

    if max_display < 1:
        raise ValueError('max_display must be positive.')

    shap_df = shap_result['shap_values']
    feature_df = shap_result['features']

    # Выбираем top-N заранее, чтобы остальные признаки не объединялись в дополнительную строку.
    feature_names = shap_df.abs().mean(axis=0).sort_values(ascending=False, kind='stable').head(max_display).index.tolist()

    # Цвет кодирует только числовые значения. Категориям не назначаем искусственный порядок.
    color_values = np.full((len(feature_df), len(feature_names)), np.nan)

    for column, feature in enumerate(feature_names):
        if pd.api.types.is_numeric_dtype(feature_df[feature]):
            color_values[:, column] = feature_df[feature].to_numpy(dtype=float)

    explanation = shap.Explanation(
        values=shap_df[feature_names].to_numpy(),
        base_values=shap_result['base_values'],
        data=color_values,
        feature_names=feature_names,
    )

    output_dir = Path(config.paths.path_to_shap)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'{config.general.experiment_name}_beeswarm.png'

    figure = plt.figure()

    try:
        shap.plots.beeswarm(
            explanation,
            max_display=len(feature_names),
            show=False,
            plot_size=(12, max(5, 0.4 * len(feature_names) + 2)),
        )

        axis = plt.gca()
        axis.set_xlabel('SHAP value — impact on log(SalePrice)')
        axis.set_title(f'OOF SHAP — {config.general.experiment_name}')

        figure = axis.figure
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(figure)

    print(f'SHAP beeswarm plot saved: {plot_path}')

    return plot_path