import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


LOG1P_FEATURES = ('LotArea', 'LotFrontage', '1stFlrSF', 'GrLivArea')


class SelectedLog1pTransformer(BaseEstimator, TransformerMixin):
    '''Apply log1p to selected right-skewed numerical features.'''

    def fit(self, features, labels=None):
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, features):
        transformed_features = features.copy()
        missing_features = [feature for feature in LOG1P_FEATURES if feature not in transformed_features]

        if missing_features:
            raise ValueError(f'Missing log1p features: {missing_features}')

        # log1p определён только для значений больше -1; выбранные площади должны быть неотрицательными.
        if (transformed_features[list(LOG1P_FEATURES)].dropna() < 0).any().any():
            raise ValueError('Selected log1p features must be non-negative.')

        for feature in LOG1P_FEATURES:
            transformed_features[feature] = np.log1p(transformed_features[feature])

        return transformed_features

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)

        return self.feature_names_in_