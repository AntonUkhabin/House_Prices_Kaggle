import sys
import traceback
from time import perf_counter

from config import config
from src import blending
from src import experiment_logging as log
from src import permutation_importance, shap_analysis
from src.data import load_data, split_train_holdout
from src.train_functions import calculate_regression_metrics, cross_validate_model_with_early_stopping, cross_validate_standard, predict_with_pipeline_ensemble
from src.utils import set_seed, setup_run_logging
from src.torch_training import cross_validate_neural_network

import warnings
# Не выводим ожидаемые предупреждения о новых категориях:
# OneHotEncoder безопасно кодирует их нулями благодаря handle_unknown='ignore'.
warnings.filterwarnings('ignore', message='Found unknown categories.*encoded as all zeros', category=UserWarning, module='sklearn.preprocessing._encoders')


def validate_analysis_config(config):
    '''Validate model interpretation settings before training.'''

    active_model = config.model.active

    if config.shap.enabled and active_model != 'catboost':
        raise ValueError(f'OOF SHAP supports only CatBoost, got: {active_model}.')

    if config.permutation_importance.enabled and active_model == 'blend':
        raise ValueError('OOF Permutation Importance is not implemented for blend. Run a blend component separately.')


def main() -> int:
    '''Run the active regression model, save predictions and capture terminal output.'''

    tee_logger = setup_run_logging(config)
    started = perf_counter()

    try:
        log.print_experiment_info(config)

        # Проверяем совместимость analysis methods до загрузки данных и обучения.
        validate_analysis_config(config)
        
        set_seed(config.general.seed)

        # Разделяем данные до обучения: holdout не участвует в CV.
        log.print_section('Data')
        
        train_df, test_df = load_data(config)
        train_cv_df, holdout_df = split_train_holdout(train_df, config)

        log.print_data_info(train_df, train_cv_df, holdout_df, test_df, config)
        log.print_model_info(config)

        # Каждый фолд обучает свой пайплайн; модели сохраняются для ансамбля.
        log.print_cv_start(config)

        if config.model.active == 'blend':
            blend_result = blending.cross_validate_blend(train_cv_df, 'SalePrice', config)
            scores = blend_result['scores']
            oof_predictions = blend_result['oof_predictions']
            fold_ids = blend_result['fold_ids']
            number_of_models = sum(len(component['fold_models']) for component in blend_result['components'].values())
        elif config.model.active == 'dnn':
            scores, fold_models, oof_predictions, fold_ids = cross_validate_neural_network(train_cv_df, 'SalePrice', config)
            number_of_models = len(fold_models)
        elif config.model.active in ('catboost', 'xgboost'):
            scores, fold_models, oof_predictions, fold_ids = cross_validate_model_with_early_stopping(train_cv_df, 'SalePrice', config)
            number_of_models = len(fold_models)
        else:
            scores, fold_models, oof_predictions, fold_ids = cross_validate_standard(train_cv_df, 'SalePrice', config)
            number_of_models = len(fold_models)

        log.print_cv_summary(scores, number_of_models)

        if config.model.active == 'blend':
            log.print_section('Blend Component Summary')

            for model_name, component in blend_result['components'].items():
                component_metrics = calculate_regression_metrics(train_cv_df['SalePrice'], component['oof_predictions'])
                mean_cv_rmse = sum(component['scores']) / len(component['scores'])
                print(f'{model_name} | weight: {component["weight"]:.3f} | mean CV RMSE(log): {mean_cv_rmse:.5f} | OOF RMSE(log): {component_metrics["rmse_log"]:.5f}')
        else:
            # Обычная диагностика рассчитана для одной active model.
            log.print_model_diagnostics(fold_models, config, top_n=20)

        if config.logging.save_training_history:
            if config.model.active == 'blend':
                for model_name, component in blend_result['components'].items():
                    component_config = component['config']
                    component_fold_models = component['fold_models']

                    if model_name in ('catboost', 'xgboost'):
                        history_path = log.save_boosting_training_history(component_fold_models, component_config)
                        log.save_boosting_learning_curves(history_path)
                    elif model_name == 'dnn':
                        history_path = log.save_dnn_training_history(component_fold_models, component_config)
                        log.save_dnn_learning_curves(history_path)
            elif config.model.active in ('catboost', 'xgboost'):
                history_path = log.save_boosting_training_history(fold_models, config)
                log.save_boosting_learning_curves(history_path)
            elif config.model.active == 'dnn':
                history_path = log.save_dnn_training_history(fold_models, config)
                log.save_dnn_learning_curves(history_path)

        # Метрики получают реальные цены в долларах и прогнозы в логарифмах.
        log.print_section('OOF Evaluation')

        oof_metrics = calculate_regression_metrics(train_cv_df['SalePrice'], oof_predictions)

        log.print_regression_metrics('OOF', oof_metrics)
        log.save_oof_predictions(train_cv_df, oof_predictions, fold_ids, config)

        if config.permutation_importance.enabled:
            log.print_section('OOF Permutation Importance')
            importance_started = perf_counter()

            importance_result = permutation_importance.calculate_oof_permutation_importance(
                train_cv_df=train_cv_df,
                target_col='SalePrice',
                fold_models=fold_models,
                fold_ids=fold_ids,
                oof_predictions=oof_predictions,
                n_repeats=config.permutation_importance.n_repeats,
                seed=config.general.seed,
            )

            importance_df = permutation_importance.save_permutation_importance(importance_result, config)

            print('\nTop 20 features by OOF permutation importance:')
            print(importance_df.head(20).to_string(index=False, float_format=lambda value: f'{value:.6f}'))
            print(f'Permutation importance runtime: {perf_counter() - importance_started:.1f} seconds')

        if config.shap.enabled:
            log.print_section('OOF SHAP Analysis')
            shap_started = perf_counter()

            shap_result = shap_analysis.calculate_oof_shap(
                train_cv_df=train_cv_df,
                target_col='SalePrice',
                fold_models=fold_models,
                fold_ids=fold_ids,
            )

            importance_df = shap_analysis.save_oof_shap(shap_result, config)
            shap_analysis.save_shap_importance_plot(importance_df, config)
            shap_analysis.save_shap_beeswarm_plot(shap_result, config)

            print(f'SHAP runtime: {perf_counter() - shap_started:.1f} seconds')

        print(f'\nHoldout evaluation enabled: {config.evaluation.evaluate_holdout}')

        if config.evaluation.evaluate_holdout:
            # Готовый ансамбль оценивается на holdout без дополнительного обучения.
            log.print_section('Holdout Evaluation')

            features_holdout = holdout_df.drop(columns=['SalePrice'])
            if config.model.active == 'blend':
                holdout_predictions_log = blending.predict_with_blend(features_holdout, blend_result)
            else:
                holdout_predictions_log = predict_with_pipeline_ensemble(features_holdout, fold_models)
            holdout_metrics = calculate_regression_metrics(holdout_df['SalePrice'], holdout_predictions_log)

            log.print_regression_metrics('Holdout', holdout_metrics)
            log.save_holdout_predictions(holdout_df, holdout_predictions_log, config)

        # Тот же ансамбль предсказывает test.csv; save_submission возвращает цены в доллары.
        log.print_submission_info(number_of_models, config)

        if config.model.active == 'blend':
            test_predictions_log = blending.predict_with_blend(test_df, blend_result)
        else:
            test_predictions_log = predict_with_pipeline_ensemble(test_df, fold_models)
        submission_df = log.save_submission(test_df, test_predictions_log, config)

        log.print_submission_summary(submission_df)

        log.print_run_summary(config, perf_counter() - started)
        return 0

    except (Exception, KeyboardInterrupt) as error:
        # Записываем traceback до закрытия лога и возвращаем код ошибки.
        log.print_section('Run Failed')
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1

    finally:
        # Восстанавливаем stdout/stderr даже при ошибке или прерывании запуска.
        try:
            tee_logger.flush()
        finally:
            sys.stdout = tee_logger.terminal
            sys.stderr = tee_logger.original_stderr
            tee_logger.close()


if __name__ == '__main__':
    sys.exit(main())
