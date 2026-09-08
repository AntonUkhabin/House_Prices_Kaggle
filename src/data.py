import pandas as pd
from omegaconf import DictConfig

from sklearn.model_selection import train_test_split


def load_data(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load train and test datasets, preserving the 'None' category.'''

    read_options = {
        'keep_default_na': False,
        'na_values': ['', 'NA'],
        'dtype': {'MSSubClass': 'object'},
    }

    train_df = pd.read_csv(config.paths.path_to_csv, **read_options)
    test_df = pd.read_csv(config.paths.path_to_kaggle_test, **read_options)

    return train_df, test_df


def remove_eda_outliers(train_cv_df: pd.DataFrame) -> pd.DataFrame:
    '''Remove anomalous observations identified during EDA.'''

    # Эти два дома имеют очень большую жилую площадь, но аномально низкую цену.
    eda_outlier_ids = {524, 1299}

    found_outlier_ids = set(train_cv_df.loc[train_cv_df['Id'].isin(eda_outlier_ids), 'Id'])

    # Защищаемся от ситуации, когда после изменения seed один из домов окажется в holdout.
    if found_outlier_ids != eda_outlier_ids:
        missing_ids = eda_outlier_ids - found_outlier_ids
        raise ValueError(f'EDA outliers are missing from Train/CV: {sorted(missing_ids)}')

    filtered_train_cv_df = train_cv_df.loc[~train_cv_df['Id'].isin(eda_outlier_ids)].copy()

    print(f'Removed EDA outliers: {sorted(eda_outlier_ids)} ({len(train_cv_df)} -> {len(filtered_train_cv_df)})')

    return filtered_train_cv_df


def split_train_holdout(train_df, config):
    '''Split labeled data into development and holdout sets.'''

    train_cv_df, holdout_df = train_test_split(
        train_df,
        test_size=config.split.test_size,
        random_state=config.general.seed,
        shuffle=True,
    )

    if config.split.remove_eda_outliers:
        train_cv_df = remove_eda_outliers(train_cv_df)

    return train_cv_df, holdout_df