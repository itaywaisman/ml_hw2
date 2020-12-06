import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline


def select_features_filter(X, y):
    relief = RFE(ReliefF(), n_features_to_select=20, step = 5, verbose=True)
    feature_selector = relief.fit(X, y['TestResultsCode'].values.codes)
    print('Selected Features:', X.columns[feature_selector.support_])
    return feature_selector


def select_features_wrapper(X,y, forward=True, k_features=20):
    # svc = SVC(gamma='auto')
    # linearSVC = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')
    random_forest_clssifier = RandomForestClassifier(max_depth=7, random_state=0)

    sgd = SGDClassifier(max_iter=1000, tol=1e-3)
#     knn = KNNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(sgd, k_features=k_features, forward=forward, floating=False,
                                    verbose=5, cv=0, n_jobs=-1)
    sfs.fit(X, y.values.ravel())
    print(sfs.k_feature_names_)
    return sfs


