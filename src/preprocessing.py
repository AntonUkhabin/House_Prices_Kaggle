import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class StructuralMissingTransformer(BaseEstimator, TransformerMixin):
    '''Replace structural missing values with meaningful categories.'''

    def fit(self, features, labels=None):
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)

        return self

    def transform(self, features):
        transformed_features = features.copy()

        # Связанные числовые features позволяют отличить отсутствие объекта от неизвестного значения.
        no_pool = transformed_features['PoolArea'].eq(0)
        no_fireplace = transformed_features['Fireplaces'].eq(0)
        no_garage = transformed_features['GarageCars'].eq(0) & transformed_features['GarageArea'].eq(0)
        no_basement = transformed_features['TotalBsmtSF'].eq(0)

        pool_missing = no_pool & transformed_features['PoolQC'].isna()
        transformed_features.loc[pool_missing, 'PoolQC'] = 'NoPool'

        fireplace_missing = no_fireplace & transformed_features['FireplaceQu'].isna()
        transformed_features.loc[fireplace_missing, 'FireplaceQu'] = 'NoFireplace'

        # Заменяем только structural missing; настоящие пропуски оставляем для imputation внутри fold.
        garage_features = [
            'GarageType',
            'GarageFinish',
            'GarageQual',
            'GarageCond',
        ]

        for feature in garage_features:
            garage_missing = no_garage & transformed_features[feature].isna()
            transformed_features.loc[garage_missing, feature] = 'NoGarage'

        basement_features = [
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
        ]

        for feature in basement_features:
            basement_missing = no_basement & transformed_features[feature].isna()
            transformed_features.loc[basement_missing, feature] = 'NoBasement'

        # Согласно описанию данных, NA в этих features непосредственно означает отсутствие объекта.
        transformed_features['Alley'] = transformed_features['Alley'].fillna('NoAlley')
        transformed_features['Fence'] = transformed_features['Fence'].fillna('NoFence')
        transformed_features['MiscFeature'] = transformed_features['MiscFeature'].fillna('NoMiscFeature')

        return transformed_features

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)

        return self.feature_names_in_


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
    '''Build preprocessing for the active model.'''

    active_model = config.model.active

    if active_model == 'linear_regression':
        return build_linear_preprocessor(config)

    if active_model == 'random_forest':
        return build_tree_preprocessor(config)

    raise ValueError(f'Unknown preprocessor for model: {active_model}')


def build_linear_preprocessor(config) -> Pipeline:
    '''Build preprocessing pipeline for linear regression models.'''

    drop_columns = list(config.preprocessing.drop_columns)

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)),
    ])

    # Селекторы определяют типы колонок во время fit после обработки structural missing.
    column_transformer = ColumnTransformer([
        ('numerical', numerical_pipeline, ColumnSelector('numerical', drop_columns)),
        ('categorical', categorical_pipeline, ColumnSelector('categorical', drop_columns)),
    ], remainder='drop')

    # Детерминированная обработка выполняется перед статистическим preprocessing внутри fold.
    return Pipeline([
        ('structural_missing', StructuralMissingTransformer()),
        ('columns', column_transformer),
    ])


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


