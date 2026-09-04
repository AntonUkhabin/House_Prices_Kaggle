import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ColumnSelector:
    '''Select features by dtype and exclude configured columns.'''

    def __init__(self, kind, drop_columns):
        self.kind = kind
        self.drop_columns = drop_columns

    def __call__(self, features):
        available = features.drop(columns=['Id', *self.drop_columns], errors='ignore')

        if self.kind == 'numerical':
            return available.select_dtypes(include=np.number).columns.tolist()

        if self.kind == 'categorical':
            return available.select_dtypes(exclude=np.number).columns.tolist()

        raise ValueError(f'Unknown feature kind: {self.kind}')


def build_preprocessor(config):
    '''Build preprocessing for the active model'''

    if config.model.active == 'random_forest':
        return build_tree_preprocessor(config)

    raise ValueError(f'Unknown preprocessor for model: {config.model.active}')


def build_tree_preprocessor(config) -> ColumnTransformer:
    '''Fill missing values and encode categorical features'''

    drop_columns = list(config.preprocessing.drop_columns)

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1, keep_empty_features=True)),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('numerical', numerical_pipeline, ColumnSelector('numerical', drop_columns)),
        ('categorical', categorical_pipeline, ColumnSelector('categorical', drop_columns)),
    ], remainder='drop')


