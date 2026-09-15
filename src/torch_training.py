from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from src.torch_models import build_torch_model
from src.train_functions import build_feature_pipeline
from src.utils import set_seed


def get_torch_device() -> torch.device:
    '''Select CUDA when available and otherwise use CPU.'''

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_data_loader(features, labels, batch_size, shuffle, seed, num_workers):
    '''Create a reproducible DataLoader for numerical and categorical features.'''

    numerical_tensor = torch.as_tensor(features['numerical'], dtype=torch.float32)
    categorical_tensor = torch.as_tensor(features['categorical'], dtype=torch.int64)

    if labels is None:
        dataset = TensorDataset(numerical_tensor, categorical_tensor)
    else:
        labels_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        dataset = TensorDataset(numerical_tensor, categorical_tensor, labels_tensor)

    # Отдельный generator делает shuffle воспроизводимым и не зависит от глобального состояния PyTorch.
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        # Pinned memory ускоряет передачу batch с CPU на CUDA; на CPU этот режим автоматически отключается.
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, data_loader, loss_function, optimizer, device, target_std):
    '''Train a regression model for one epoch and return RMSE on the log-target scale.'''

    model.train()

    squared_error_sum = 0.0
    sample_count = 0

    for numerical_features, categorical_features, targets in data_loader:
        numerical_features = numerical_features.to(device, non_blocking=True)
        categorical_features = categorical_features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(numerical_features, categorical_features)
        loss = loss_function(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        squared_error_sum += loss.item() * batch_size
        sample_count += batch_size

    rmse_scaled = np.sqrt(squared_error_sum / sample_count)
    return float(rmse_scaled * target_std)


def validate_one_epoch(model, data_loader, loss_function, device, target_std):
    '''Evaluate a regression model and return RMSE on the log-target scale.'''

    model.eval()

    squared_error_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for numerical_features, categorical_features, targets in data_loader:
            numerical_features = numerical_features.to(device, non_blocking=True)
            categorical_features = categorical_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(numerical_features, categorical_features)
            loss = loss_function(predictions, targets)

            # Учитываем размер batch, чтобы последний неполный batch не получил такой же вес, как полный.
            batch_size = targets.size(0)
            squared_error_sum += loss.item() * batch_size
            sample_count += batch_size

    # Возвращаем RMSE из standardized target обратно в масштаб log(SalePrice).
    rmse_scaled = np.sqrt(squared_error_sum / sample_count)
    return float(rmse_scaled * target_std)


def train_fold(model, train_loader, val_loader, loss_function, optimizer, device, target_std, epochs, early_stopping_rounds, min_delta, checkpoint_path, fold, log_interval):
    '''Train one regression fold with early stopping and restore its best checkpoint.'''

    if epochs < 1:
        raise ValueError('epochs must be positive.')

    if early_stopping_rounds < 1:
        raise ValueError('early_stopping_rounds must be positive.')

    if log_interval < 1:
        raise ValueError('log_interval must be positive.')

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        'train_rmse': [],
        'validation_rmse': [],
        'learning_rate': [],
    }

    best_validation_rmse = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, loss_function, optimizer, device, target_std)
        validation_rmse = validate_one_epoch(model, val_loader, loss_function, device, target_std)
        learning_rate = optimizer.param_groups[0]['lr']

        if not np.isfinite(train_rmse) or not np.isfinite(validation_rmse):
            raise RuntimeError(f'Non-finite RMSE detected on fold {fold}, epoch {epoch}.')

        history['train_rmse'].append(train_rmse)
        history['validation_rmse'].append(validation_rmse)
        history['learning_rate'].append(float(learning_rate))

        improved = validation_rmse < best_validation_rmse - min_delta

        if improved:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            epochs_without_improvement = 0

            # Сохраняем веса лучшей по validation RMSE эпохи, а не последней выполненной эпохи.
            torch.save({
                'fold': fold,
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_validation_rmse': best_validation_rmse,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        should_stop = epochs_without_improvement >= early_stopping_rounds
        should_log = epoch == 1 or epoch % log_interval == 0 or epoch == epochs or should_stop
        improvement_marker = ' *' if improved else ''

        if should_log:
            print(f'Fold {fold} | Epoch {epoch:03d}/{epochs} | RMSE(log): {train_rmse:.5f}/{validation_rmse:.5f} | lr: {learning_rate:.6f}{improvement_marker}')

        if should_stop:
            print(f'Fold {fold} | Early stopping on epoch {epoch}')
            break

    if best_epoch == 0:
        raise RuntimeError(f'No checkpoint was saved for fold {fold}.')

    # После early stopping восстанавливаем веса лучшей эпохи для последующих predictions.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Fold {fold} | Best epoch: {best_epoch} | Best validation RMSE(log): {best_validation_rmse:.5f}')

    return model, history, best_epoch, best_validation_rmse


def predict_standardized(model, data_loader, device):
    '''Predict standardized targets with a PyTorch regression model.'''

    model.eval()
    predictions = []

    with torch.no_grad():
        for numerical_features, categorical_features in data_loader:
            numerical_features = numerical_features.to(device, non_blocking=True)
            categorical_features = categorical_features.to(device, non_blocking=True)

            batch_predictions = model(numerical_features, categorical_features)
            # Переносим каждый batch predictions на CPU, чтобы результаты не накапливались в GPU memory.
            predictions.append(batch_predictions.cpu())

    if not predictions:
        return np.empty(0, dtype=np.float32)

    return torch.cat(predictions).numpy()


class DNNFoldModel:
    '''Store one fitted DNN fold and expose log-scale predictions.'''

    def __init__(self, preprocessor, model, target_mean, target_std, batch_size, seed, num_workers, fold, history, best_epoch, best_validation_rmse):
        if target_std <= 0:
            raise ValueError('target_std must be positive.')

        self.preprocessor = preprocessor
        # Храним fold-модель на CPU и переносим на accelerator только на время prediction.
        self.model = model.to('cpu')
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.fold = int(fold)
        self.history = history
        self.best_epoch = int(best_epoch)
        self.best_validation_rmse = float(best_validation_rmse)

    def predict(self, features):
        '''Predict log prices from raw input features.'''

        transformed_features = self.preprocessor.transform(features)
        data_loader = create_data_loader(transformed_features, None, self.batch_size, False, self.seed, self.num_workers)
        device = get_torch_device()
        self.model.to(device)

        try:
            predictions_scaled = predict_standardized(self.model, data_loader, device)
        finally:
            # Fold-модель возвращаем на CPU, чтобы ансамбль не хранил все модели в GPU memory.
            self.model.to('cpu')

        # Преобразуем standardized prediction обратно в log(SalePrice), сохраняя общий интерфейс моделей.
        predictions_log = predictions_scaled * self.target_std + self.target_mean

        if not np.isfinite(predictions_log).all():
            raise RuntimeError(f'Non-finite predictions produced by DNN fold {self.fold}.')

        return predictions_log


def cross_validate_neural_network(train_cv_df, target_col, config):
    '''Run fold-specific DNN training and generate out-of-fold log predictions.'''

    features = train_cv_df.drop(columns=[target_col])
    labels_log = np.log(train_cv_df[target_col].to_numpy(dtype=np.float64))

    kfold = KFold(n_splits=config.split.n_splits, shuffle=config.dataloader_params.shuffle, random_state=config.training.fold_seed)

    model_params = config.model.models.dnn
    device = get_torch_device()

    print(f'PyTorch device: {device}')

    scores = []
    fold_models = []
    # Внутри архитектуры fold_ids хранятся как 0..n-1, а при сохранении в CSV преобразуются в 1..n.
    oof_predictions = np.full(len(features), np.nan, dtype=float)
    fold_ids = np.full(len(features), -1, dtype=np.int16)

    # Разделяем внутренний zero-based индекс и отображаемый пользователю номер fold.
    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(features)):
        fold = fold_index + 1
        print(f'\nFold {fold}')

        # Сбрасываем random state перед каждым fold для полной воспроизводимости CV.
        set_seed(config.general.seed)

        features_train = features.iloc[train_idx]
        features_val = features.iloc[val_idx]
        labels_train = labels_log[train_idx]
        labels_val = labels_log[val_idx]

        # Все статистические преобразования и vocabulary обучаются только на train fold.
        feature_pipeline = build_feature_pipeline(config)
        features_train_transformed = feature_pipeline.fit_transform(features_train, labels_train)
        features_val_transformed = feature_pipeline.transform(features_val)

        # Стандартизуем target статистиками только train fold для стабильного обучения без data leakage.
        target_mean = float(labels_train.mean())
        target_std = float(labels_train.std())

        if not np.isfinite(target_std) or target_std <= 0:
            raise ValueError(f'Invalid target standard deviation on fold {fold}: {target_std}.')

        labels_train_scaled = (labels_train - target_mean) / target_std
        labels_val_scaled = (labels_val - target_mean) / target_std

        dnn_preprocessor = feature_pipeline.named_steps['preprocessor'].named_steps['dnn_features']

        train_loader = create_data_loader(features_train_transformed, labels_train_scaled, model_params.batch_size, True, config.general.seed, model_params.num_workers)
        val_loader = create_data_loader(features_val_transformed, labels_val_scaled, model_params.batch_size, False, config.general.seed, model_params.num_workers)

        # Размеры Embedding зависят от vocabulary текущего train fold и могут немного различаться между фолдами.
        model = build_torch_model(config.model.active, features_train_transformed['numerical'].shape[1], dnn_preprocessor.category_cardinalities_).to(device)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_params.learning_rate, weight_decay=model_params.weight_decay)

        checkpoint_path = Path(config.paths.path_to_checkpoints) / config.general.experiment_name / f'fold_{fold}.pt'

        model, history, best_epoch, best_validation_rmse = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            target_std=target_std,
            epochs=model_params.epochs,
            early_stopping_rounds=model_params.early_stopping_rounds,
            min_delta=model_params.min_delta,
            checkpoint_path=checkpoint_path,
            fold=fold,
            log_interval=config.logging.training_log_interval,
        )

        fold_model = DNNFoldModel(
            preprocessor=feature_pipeline,
            model=model,
            target_mean=target_mean,
            target_std=target_std,
            batch_size=model_params.batch_size,
            seed=config.general.seed,
            num_workers=model_params.num_workers,
            fold=fold,
            history=history,
            best_epoch=best_epoch,
            best_validation_rmse=best_validation_rmse,
        )

        validation_predictions = fold_model.predict(features_val)
        fold_score = root_mean_squared_error(labels_val, validation_predictions)

        if validation_predictions.shape != (len(val_idx),):
            raise ValueError(f'Unexpected prediction shape on fold {fold}: {validation_predictions.shape}.')

        if not np.isclose(fold_score, best_validation_rmse, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f'Restored DNN score differs from the best validation score on fold {fold}.')

        # Записываем каждой строке prediction модели, которая не обучалась на этой строке.
        oof_predictions[val_idx] = validation_predictions
        fold_ids[val_idx] = fold_index
        scores.append(float(fold_score))
        fold_models.append(fold_model)

        print(f'Fold {fold} | RMSE(log): {fold_score:.5f} | Best epoch: {best_epoch} | Epochs run: {len(history["validation_rmse"])}')

    if np.isnan(oof_predictions).any() or (fold_ids < 0).any():
        raise RuntimeError('OOF predictions or fold identifiers are incomplete.')

    return scores, fold_models, oof_predictions, fold_ids