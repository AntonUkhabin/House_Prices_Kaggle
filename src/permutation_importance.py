from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error


def calculate_oof_permutation_importance(train_cv_df, target_col, fold_models, fold_ids, oof_predictions, n_repeats, seed):
    '''Calculate raw-feature OOF permutation importance for fitted fold models.'''

    if not fold_models:
        raise ValueError('No fitted fold models provided.')

    if n_repeats < 1:
        raise ValueError('n_repeats must be positive.')

    # Любая fold model получает raw DataFrame и самостоятельно применяет свой preprocessing.
    features = train_cv_df.drop(columns=[target_col])
    feature_names = [feature for feature in features.columns if feature != 'Id']
    labels_log = np.log(train_cv_df[target_col].to_numpy(dtype=np.float64))
    fold_ids = np.asarray(fold_ids)
    oof_predictions = np.asarray(oof_predictions, dtype=float)

    if fold_ids.shape != (len(features),):
        raise ValueError('Expected one fold ID per development row.')

    if oof_predictions.shape != (len(features),):
        raise ValueError('Expected one OOF prediction per development row.')

    if not np.isfinite(oof_predictions).all():
        raise ValueError('OOF predictions must be finite.')

    expected_fold_ids = np.arange(len(fold_models))

    if not np.array_equal(np.unique(fold_ids), expected_fold_ids):
        raise ValueError('Fold IDs must match the supplied fold models.')

    missing_features = [feature for feature in feature_names if feature not in features]

    # Рассчитываем baseline RMSE для полного OOF и каждого fold.
    baseline_oof_rmse = root_mean_squared_error(labels_log, oof_predictions)
    baseline_fold_rmse = {
        fold: root_mean_squared_error(labels_log[fold_ids == fold], oof_predictions[fold_ids == fold])
        for fold in expected_fold_ids
    }

    rng = np.random.default_rng(seed)
    repeat_records = []
    fold_records = []

    for feature_number, feature in enumerate(feature_names, start=1):
        for repeat in range(1, n_repeats + 1):
            permuted_oof_predictions = oof_predictions.copy()

            for fold, fold_model in enumerate(fold_models):
                val_idx = np.flatnonzero(fold_ids == fold)
                features_val = features.iloc[val_idx].copy()

                # Перемешиваем один raw feature только внутри соответствующего validation fold.
                feature_values = features_val[feature].to_numpy(copy=True)
                features_val[feature] = feature_values[rng.permutation(len(feature_values))]

                # Pipeline или DNNFoldModel самостоятельно применяет fitted preprocessing.
                permuted_predictions = np.asarray(fold_model.predict(features_val), dtype=float).reshape(-1)

                if permuted_predictions.shape != (len(val_idx),):
                    raise ValueError(f'Unexpected prediction shape for feature {feature}, fold {fold + 1}: {permuted_predictions.shape}.')

                if not np.isfinite(permuted_predictions).all():
                    raise ValueError(f'Non-finite predictions for feature {feature}, fold {fold + 1}.')

                permuted_oof_predictions[val_idx] = permuted_predictions
                permuted_fold_rmse = root_mean_squared_error(labels_log[val_idx], permuted_predictions)

                fold_records.append({
                    'feature': feature,
                    'repeat': repeat,
                    'fold': fold + 1,
                    'importance': permuted_fold_rmse - baseline_fold_rmse[fold],
                })

            # Положительный importance означает ухудшение OOF RMSE после permutation.
            permuted_oof_rmse = root_mean_squared_error(labels_log, permuted_oof_predictions)

            repeat_records.append({
                'feature': feature,
                'repeat': repeat,
                'importance': permuted_oof_rmse - baseline_oof_rmse,
            })

        if feature_number == 1 or feature_number % 10 == 0 or feature_number == len(feature_names):
            print(f'Permutation importance: {feature_number}/{len(feature_names)} features')

    repeat_df = pd.DataFrame(repeat_records)
    fold_df = pd.DataFrame(fold_records)

    # Усредняем OOF importance по независимым permutations.
    summary_df = repeat_df.groupby('feature', sort=False).agg(
        importance_mean=('importance', 'mean'),
        importance_std=('importance', 'std'),
        importance_min=('importance', 'min'),
        importance_max=('importance', 'max'),
        positive_repeat_fraction=('importance', lambda values: np.mean(values > 0)),
    ).reset_index()

    # Сначала усредняем repeats внутри fold, затем считаем долю folds с положительным importance.
    fold_mean_df = fold_df.groupby(['feature', 'fold'], sort=False)['importance'].mean().reset_index()
    fold_stability_df = fold_mean_df.groupby('feature', sort=False).agg(
        positive_fold_fraction=('importance', lambda values: np.mean(values > 0)),
    ).reset_index()

    summary_df = summary_df.merge(fold_stability_df, on='feature', validate='one_to_one')
    summary_df['importance_std'] = summary_df['importance_std'].fillna(0.0)
    summary_df = summary_df.sort_values('importance_mean', ascending=False, kind='stable').reset_index(drop=True)

    return {
        'baseline_oof_rmse': float(baseline_oof_rmse),
        'summary': summary_df,
        'repeats': repeat_df,
        'folds': fold_df,
    }


def save_permutation_importance(result, config) -> pd.DataFrame:
    '''Save the OOF permutation importance summary.'''

    summary_df = result['summary']

    if summary_df.empty:
        raise ValueError('Permutation importance summary is empty.')

    output_dir = Path(config.paths.path_to_permutation_importance)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{config.general.experiment_name}_permutation_importance.csv'

    # Подробные repeats и folds оставляем в памяти, а на диск сохраняем итоговый summary.
    summary_df.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')

    print(f'Baseline OOF RMSE(log): {result["baseline_oof_rmse"]:.5f}')
    print(f'Permutation importance saved: {output_path}')

    return summary_df