import sys
import traceback
from time import perf_counter

from config import config
from src import experiment_logging as log
from src.data import load_data, split_train_holdout
from src.train_functions import calculate_regression_metrics, cross_validate_model_with_early_stopping, cross_validate_standard, predict_with_pipeline_ensemble
from src.utils import set_seed, setup_run_logging

import warnings
# Не выводим ожидаемые предупреждения о новых категориях:
# OneHotEncoder безопасно кодирует их нулями благодаря handle_unknown='ignore'.
warnings.filterwarnings('ignore', message='Found unknown categories.*encoded as all zeros', category=UserWarning, module='sklearn.preprocessing._encoders')


def main() -> int:
    '''Run the active regression model, save predictions and capture terminal output.'''

    tee_logger = setup_run_logging(config)
    started = perf_counter()

    try:
        log.print_experiment_info(config)
        
        set_seed(config.general.seed)

        # Разделяем данные до обучения: holdout не участвует в CV.
        log.print_section('Data')
        
        train_df, test_df = load_data(config)
        train_cv_df, holdout_df = split_train_holdout(train_df, config)

        log.print_data_info(train_df, train_cv_df, holdout_df, test_df, config)
        log.print_model_info(config)

        # Каждый фолд обучает свой пайплайн; модели сохраняются для ансамбля.
        log.print_cv_start(config)

        if config.model.active == 'catboost':
            scores, fold_models, oof_predictions, fold_ids = cross_validate_model_with_early_stopping(train_cv_df, 'SalePrice', config)
        else:
            scores, fold_models, oof_predictions, fold_ids = cross_validate_standard(train_cv_df, 'SalePrice', config)

        log.print_cv_summary(scores, len(fold_models))

        # Выводим диагностику, соответствующую активной модели.
        log.print_model_diagnostics(fold_models, config, top_n=20)

        # Метрики получают реальные цены в долларах и прогнозы в логарифмах.
        log.print_section('OOF Evaluation')

        oof_metrics = calculate_regression_metrics(train_cv_df['SalePrice'], oof_predictions)

        log.print_regression_metrics('OOF', oof_metrics)
        log.save_oof_predictions(train_cv_df, oof_predictions, fold_ids, config)

        print(f'\nHoldout evaluation enabled: {config.evaluation.evaluate_holdout}')

        if config.evaluation.evaluate_holdout:
            # Готовый ансамбль оценивается на holdout без дополнительного обучения.
            log.print_section('Holdout Evaluation')

            features_holdout = holdout_df.drop(columns=['SalePrice'])
            holdout_predictions_log = predict_with_pipeline_ensemble(features_holdout, fold_models)
            holdout_metrics = calculate_regression_metrics(holdout_df['SalePrice'], holdout_predictions_log)

            log.print_regression_metrics('Holdout', holdout_metrics)
            log.save_holdout_predictions(holdout_df, holdout_predictions_log, config)

        # Тот же ансамбль предсказывает test.csv; save_submission возвращает цены в доллары.
        log.print_submission_info(len(fold_models))

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
