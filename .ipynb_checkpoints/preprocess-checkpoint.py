import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from globals import numeric_features, positive_scaled_features, negative_scaled_features


class Imputer:
    def __init__(self):
        self.iterative_imputer = IterativeImputer(initial_strategy='median')
        self.knn_imputer = KNNImputer(n_neighbors=5)

    def fit(self, X, y, **kargs):
        self.iterative_imputer.fit(X[numeric_features])
        self.knn_imputer.fit(X)
        # print(X.shape)
        return self

    def transform(self, X):
        X[numeric_features] = self.iterative_imputer.transform(X[numeric_features])
        res = self.knn_imputer.transform(X)
        X = pd.DataFrame(res, columns=X.columns)
        return X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


class OutlierClipper:
    def __init__(self, features):
        self._features = features
        self._feature_map = {}

    def fit(self, X, y, **kwargs):
        df = X[self._features]
        features = list(df.columns)
        for feature in features:
            f_q1 = df[feature].quantile(0.25)
            f_q3 = df[feature].quantile(0.75)
            f_iqr = f_q3 - f_q1
            self._feature_map[feature] = (f_q1 - (1.5 * f_iqr), f_q3 + (1.5 * f_iqr))
        return self

    def transform(self, data):
        data_copy = data.copy()
        for feature in self._feature_map.keys():
            data_copy[feature] = data_copy[feature].clip(lower=self._feature_map[feature][0],
                                                         upper=self._feature_map[feature][1])
        return data_copy

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


class Normalizer:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        self.max_abs_scaler = MaxAbsScaler()

    def fit(self, X, y, **kargs):
        self.min_max_scaler.fit(X[positive_scaled_features])
        self.max_abs_scaler.fit(X[negative_scaled_features])

        return self

    def transform(self, X, **kargs):
        X[positive_scaled_features] = self.min_max_scaler.transform(X[positive_scaled_features])
        X[negative_scaled_features] = self.max_abs_scaler.transform(X[negative_scaled_features])

        X['CurrentLocation_Lat'] /= 180
        X['CurrentLocation_Long'] /= 180

        return X


