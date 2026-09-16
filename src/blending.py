import numpy as np
import pandas as pd

from copy import deepcopy

from sklearn.metrics import root_mean_squared_error

from src.torch_training import cross_validate_neural_network
from src.train_functions import cross_validate_model_with_early_stopping, cross_validate_standard, predict_with_pipeline_ensemble


def validate_blend_weights(weights):
    '''Validate model weights used for blending.'''

    if not weights:
        raise ValueError('Blend weights must not be empty.')

    weight_values = np.asarray(list(weights.values()), dtype=float)

    if not np.isfinite(weight_values).all():
        raise ValueError('Blend weights must be finite.')

    if (weight_values < 0).any():
        raise ValueError('Blend weights must be non-negative.')

    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError(f'Blend weights must sum to 1.0, got {weight_values.sum():.10f}.')


def align_oof_frames(oof_frames):
    '''Validate and align OOF prediction frames by Id.'''

    if not oof_frames:
        raise ValueError('OOF prediction frames must not be empty.')

    required_columns = {'Id', 'SalePrice', 'fold', 'actual_log', 'prediction_log'}
    aligned_frames = {}

    for model_name, frame in oof_frames.items():
        missing_columns = required_columns.difference(frame.columns)

        if missing_columns:
            raise ValueError(f'Missing OOF columns for {model_name}: {sorted(missing_columns)}')

        if frame['Id'].duplicated().any():
            raise ValueError(f'Duplicate OOF Id values found for {model_name}.')

        # Сортируем все OOF frames по Id перед сравнением и blending.
        aligned_frame = frame.sort_values('Id').reset_index(drop=True).copy()

        if not np.isfinite(aligned_frame['prediction_log']).all():
            raise ValueError(f'Non-finite OOF predictions found for {model_name}.')

        aligned_frames[model_name] = aligned_frame

    reference_name = next(iter(aligned_frames))
    reference_frame = aligned_frames[reference_name]

    if not np.allclose(reference_frame['actual_log'], np.log(reference_frame['SalePrice']), rtol=1e-10, atol=1e-10):
        raise ValueError(f'actual_log does not match SalePrice for {reference_name}.')

    for model_name, frame in aligned_frames.items():
        if len(frame) != len(reference_frame):
            raise ValueError(f'OOF row count differs for {model_name}.')

        if not frame['Id'].equals(reference_frame['Id']):
            raise ValueError(f'OOF Id values differ for {model_name}.')

        if not frame['fold'].equals(reference_frame['fold']):
            raise ValueError(f'OOF fold assignments differ for {model_name}.')

        if not np.allclose(frame['SalePrice'], reference_frame['SalePrice']):
            raise ValueError(f'OOF SalePrice values differ for {model_name}.')

        if not np.allclose(frame['actual_log'], reference_frame['actual_log']):
            raise ValueError(f'OOF actual_log values differ for {model_name}.')

    return aligned_frames


def blend_oof_predictions(oof_frames, weights):
    '''Blend aligned OOF predictions in log space.'''

    validate_blend_weights(weights)

    if set(oof_frames) != set(weights):
        raise ValueError('OOF model names must match blend weight names.')

    aligned_frames = align_oof_frames(oof_frames)
    reference_name = next(iter(aligned_frames))
    blend_df = aligned_frames[reference_name].copy()

    # Смешиваем predictions в log-space, соответствующем основной метрике RMSLE.
    blended_predictions_log = sum(weights[model_name] * aligned_frames[model_name]['prediction_log'].to_numpy() for model_name in weights)
    blended_predictions_price = np.exp(blended_predictions_log)

    blend_df['prediction_log'] = blended_predictions_log
    blend_df['prediction_price'] = blended_predictions_price
    blend_df['error_log'] = blend_df['prediction_log'] - blend_df['actual_log']
    blend_df['abs_error_log'] = blend_df['error_log'].abs()
    blend_df['error_price'] = blend_df['prediction_price'] - blend_df['SalePrice']

    return blend_df


def get_component_cv_function(model_name):
    '''Return the cross-validation function for a blend component.'''

    if model_name == 'dnn':
        return cross_validate_neural_network

    if model_name in ('catboost', 'xgboost'):
        return cross_validate_model_with_early_stopping

    return cross_validate_standard


def cross_validate_blend(train_cv_df, target_col, config):
    '''Train blend components and combine their OOF predictions.'''

    weights = dict(config.blending.weights)
    validate_blend_weights(weights)

    unknown_models = [model_name for model_name in weights if model_name not in config.model.models]

    if unknown_models:
        raise ValueError(f'Unknown blend models: {unknown_models}')

    labels_log = np.log(train_cv_df[target_col].to_numpy(dtype=np.float64))
    blended_oof_predictions = np.zeros(len(train_cv_df), dtype=float)
    reference_fold_ids = None
    component_results = {}

    for model_name, weight in weights.items():
        print(f'\nBlend component: {model_name} | Weight: {weight:.3f}')

        # Каждый component получает независимую копию config со своей active model.
        component_config = deepcopy(config)
        component_config.model.active = model_name
        component_config.general.experiment_name = f'{config.general.experiment_name}_{model_name}'

        cv_function = get_component_cv_function(model_name)
        scores, fold_models, oof_predictions, fold_ids = cv_function(train_cv_df, target_col, component_config)

        oof_predictions = np.asarray(oof_predictions, dtype=float)
        fold_ids = np.asarray(fold_ids)

        if oof_predictions.shape != (len(train_cv_df),):
            raise ValueError(f'Unexpected OOF prediction shape for {model_name}: {oof_predictions.shape}.')

        if not np.isfinite(oof_predictions).all():
            raise ValueError(f'Non-finite OOF predictions produced by {model_name}.')

        # Все components должны использовать одинаковые validation rows в каждом fold.
        if reference_fold_ids is None:
            reference_fold_ids = fold_ids.copy()
        elif not np.array_equal(fold_ids, reference_fold_ids):
            raise ValueError(f'Fold assignments differ for blend component: {model_name}.')

        blended_oof_predictions += weight * oof_predictions

        component_results[model_name] = {
            'weight': float(weight),
            'config': component_config,
            'scores': scores,
            'fold_models': fold_models,
            'oof_predictions': oof_predictions,
        }

        component_rmse = root_mean_squared_error(labels_log, oof_predictions)
        print(f'Blend component completed: {model_name} | OOF RMSE(log): {component_rmse:.5f}')

    if reference_fold_ids is None:
        raise RuntimeError('Blend did not train any components.')

    if not np.isfinite(blended_oof_predictions).all():
        raise RuntimeError('Blend produced non-finite OOF predictions.')

    # Рассчитываем fold scores уже для итогового blended OOF prediction.
    blend_scores = []

    for fold in np.unique(reference_fold_ids):
        fold_mask = reference_fold_ids == fold
        fold_rmse = root_mean_squared_error(labels_log[fold_mask], blended_oof_predictions[fold_mask])
        blend_scores.append(float(fold_rmse))
        print(f'Blend Fold {fold + 1}: RMSE(log) = {fold_rmse:.5f}')

    blend_rmse = root_mean_squared_error(labels_log, blended_oof_predictions)
    print(f'Blend OOF RMSE(log): {blend_rmse:.5f}')

    return {
        'weights': weights,
        'scores': blend_scores,
        'components': component_results,
        'oof_predictions': blended_oof_predictions,
        'fold_ids': reference_fold_ids,
    }


def predict_with_blend(features, blend_result):
    '''Predict with every blend component and combine log predictions.'''

    weights = blend_result['weights']
    component_results = blend_result['components']

    validate_blend_weights(weights)

    if set(weights) != set(component_results):
        raise ValueError('Blend component names do not match blend weights.')

    blended_predictions = np.zeros(len(features), dtype=float)

    for model_name, weight in weights.items():
        fold_models = component_results[model_name]['fold_models']

        if not fold_models:
            raise ValueError(f'No fitted fold models found for blend component: {model_name}.')

        # Сначала усредняем predictions всех fold models одного component.
        component_predictions = predict_with_pipeline_ensemble(features, fold_models)
        component_predictions = np.asarray(component_predictions, dtype=float)

        if component_predictions.shape != (len(features),):
            raise ValueError(f'Unexpected prediction shape for {model_name}: {component_predictions.shape}.')

        if not np.isfinite(component_predictions).all():
            raise ValueError(f'Non-finite predictions produced by blend component: {model_name}.')

        # Затем применяем model-level weight к усреднённому component prediction.
        blended_predictions += weight * component_predictions

    if not np.isfinite(blended_predictions).all():
        raise RuntimeError('Blend produced non-finite predictions.')

    return blended_predictions