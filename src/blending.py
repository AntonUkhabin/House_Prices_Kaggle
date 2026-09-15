import numpy as np
import pandas as pd


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