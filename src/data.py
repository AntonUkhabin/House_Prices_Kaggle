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


def split_train_holdout(train_df, config):
    '''Split labeled data into development and holdout sets.'''

    train_cv_df, holdout_df = train_test_split(
        train_df,
        test_size=config.split.test_size,
        random_state=config.general.seed,
        shuffle=True,
    )

    return train_cv_df, holdout_df