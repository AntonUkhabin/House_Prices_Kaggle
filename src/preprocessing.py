import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


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


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    '''Prepare a DataFrame for native CatBoost processing.'''

    def __init__(self, drop_columns=()):
        self.drop_columns = drop_columns

    def fit(self, features, labels=None):
        '''Store selected columns and categorical feature names.'''

        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        selected_features = features.drop(columns=['Id', *self.drop_columns], errors='ignore')

        # Фиксируем состав, порядок и типы признаков по train, чтобы validation и inference использовали ту же схему.
        self.selected_features_ = selected_features.columns.tolist()
        self.categorical_features_ = selected_features.select_dtypes(exclude=np.number).columns.tolist()

        return self

    def transform(self, features):
        '''Select fitted columns and normalize categorical values.'''

        check_is_fitted(self, ['selected_features_', 'categorical_features_'])
        transformed_features = features.loc[:, self.selected_features_].copy()

        # Числовые NaN оставляем CatBoost; категориальные пропуски заменяем до преобразования в строки.
        for feature in self.categorical_features_:
            transformed_features[feature] = transformed_features[feature].astype('object').fillna('Unknown').astype(str)

        return transformed_features

    def get_feature_names_out(self, input_features=None):
        '''Return selected feature names in transformation order.'''

        check_is_fitted(self, 'selected_features_')
        return np.asarray(self.selected_features_, dtype=object)


class DNNPreprocessor(BaseEstimator, TransformerMixin):
    '''Prepare numerical features and categorical indices for a tabular DNN.'''

    def __init__(self, drop_columns=()):
        self.drop_columns = drop_columns

    def fit(self, features, labels=None):
        '''Fit numerical transformations and categorical vocabularies.'''

        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        selected_features = features.drop(columns=['Id', *self.drop_columns], errors='ignore')

        self.selected_features_ = selected_features.columns.tolist()
        self.numerical_features_ = selected_features.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features_ = selected_features.select_dtypes(exclude=np.number).columns.tolist()

        self.numerical_imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        numerical_imputed = self.numerical_imputer_.fit_transform(selected_features[self.numerical_features_])

        self.numerical_scaler_ = StandardScaler()
        self.numerical_scaler_.fit(numerical_imputed)

        self.category_maps_ = {}

        for feature in self.categorical_features_:
            normalized = self._normalize_categories(selected_features[feature])

            # Индекс 0 зарезервирован одновременно для missing и unseen categories.
            known_categories = sorted(category for category in normalized.unique() if category != 'Unknown')
            self.category_maps_[feature] = {
                category: index
                for index, category in enumerate(['Unknown', *known_categories])
            }

        self.category_cardinalities_ = [
            len(self.category_maps_[feature])
            for feature in self.categorical_features_
        ]

        return self

    def transform(self, features):
        '''Transform features into numerical values and categorical indices.'''

        check_is_fitted(self, ['selected_features_', 'numerical_features_', 'categorical_features_', 'numerical_imputer_', 'numerical_scaler_', 
                'category_maps_', 'category_cardinalities_'])

        selected_features = features.loc[:, self.selected_features_]

        numerical_features = self.numerical_imputer_.transform(selected_features[self.numerical_features_])
        numerical_features = self.numerical_scaler_.transform(numerical_features).astype(np.float32)

        categorical_columns = []

        for feature in self.categorical_features_:
            normalized = self._normalize_categories(selected_features[feature])
            category_map = self.category_maps_[feature]

            # Новые категории validation/test переводятся в зарезервированный индекс 0.
            encoded = normalized.map(category_map).fillna(0).to_numpy(dtype=np.int64)
            categorical_columns.append(encoded)

        if categorical_columns:
            categorical_features = np.column_stack(categorical_columns)
        else:
            categorical_features = np.empty((len(selected_features), 0), dtype=np.int64)

        return {
            'numerical': numerical_features,
            'categorical': categorical_features,
        }

    def get_feature_names_out(self, input_features=None):
        '''Return selected input feature names.'''

        check_is_fitted(self, 'selected_features_')
        return np.asarray(self.selected_features_, dtype=object)

    @staticmethod
    def _normalize_categories(feature):
        '''Convert missing and categorical values into stable strings.'''

        return feature.astype('object').where(feature.notna(), 'Unknown').astype(str)


def build_preprocessor(config):
    '''Build preprocessing for the active model.'''

    active_model = config.model.active

    if active_model in ('linear_regression', 'ridge', 'lasso', 'elastic_net'):
        return build_linear_preprocessor(config)

    if active_model == 'random_forest':
        return build_tree_preprocessor(config)

    if active_model == 'catboost':
        return build_catboost_preprocessor(config)

    if active_model == 'xgboost':
        return build_xgboost_preprocessor(config)

    if active_model == 'knn':
        return build_knn_preprocessor(config)

    if active_model == 'dnn':
        return build_dnn_preprocessor(config)

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


def build_catboost_preprocessor(config) -> Pipeline:
    '''Build preprocessing with structural missing categories for CatBoost.'''

    return Pipeline([
        # Structural missing обрабатываем до удаления колонок: некоторые из них нужны для определения отсутствия объекта.
        ('structural_missing', StructuralMissingTransformer()),
        ('catboost_features', CatBoostPreprocessor(drop_columns=list(config.preprocessing.drop_columns))),
    ])


def build_xgboost_preprocessor(config) -> Pipeline:
    '''Build XGBoost preprocessing with native numerical missing values and one-hot categories.'''

    drop_columns = list(config.preprocessing.drop_columns)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    column_transformer = ColumnTransformer([
        ('numerical', 'passthrough', ColumnSelector('numerical', drop_columns)),
        ('categorical', categorical_pipeline, ColumnSelector('categorical', drop_columns)),
    ], remainder='drop')

    return Pipeline([
        # Structural missing определяем до удаления колонок, нужных для проверки отсутствия объекта.
        ('structural_missing', StructuralMissingTransformer()),
        ('columns', column_transformer),
    ])


def build_knn_preprocessor(config) -> ColumnTransformer:
    '''Build model-specific preprocessing for selected KNN features.'''

    selected_features = list(config.preprocessing.knn_features)
    ordinal_features = list(config.preprocessing.knn_ordinal_features)
    nominal_features = list(config.preprocessing.knn_nominal_features)
    categorical_features = ordinal_features + nominal_features
    numerical_features = [feature for feature in selected_features if feature not in categorical_features]

    quality_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        # Сохраняем естественный порядок quality features перед расчётом расстояний.
        ('encoder', OrdinalEncoder(
            categories=[quality_order] * len(ordinal_features),
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
        ('scaler', StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        # Незнакомый район при inference получает нули во всех обученных OHE-колонках.
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('nominal', nominal_pipeline, nominal_features),
    ], remainder='drop', verbose_feature_names_out=False)


def build_dnn_preprocessor(config) -> Pipeline:
    '''Build fold-specific preprocessing for a tabular DNN with embeddings.'''

    return Pipeline([
        ('structural_missing', StructuralMissingTransformer()),
        ('dnn_features', DNNPreprocessor(drop_columns=list(config.preprocessing.dnn_drop_columns))),
    ])