import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        ind = np.random.choice(X.shape[0], 2000, replace=False)
        ind1 = ind[:1000]
        ind2 = ind[1000:]
        d = []
        for i in ind1:
            for j in ind2:
                if i != j:
                    d.append(np.sum(X[i] - X[j]))
        d = np.array(d)**2
        sigma = np.median(d)
        self.w = np.random.normal(0, 1 / np.sqrt(sigma), (X.shape[1], self.n_features))
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        return self.func(np.dot(X, self.w) + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        super().fit(X, y)
        if X.shape[1] <= self.n_features:
            Q, R = np.linalg.qr(self.w.T, mode='reduced')
            self.w = (2 / self.n_features) * Q.T
        else:
            Q, R = np.linalg.qr(self.w, mode='reduced')
            self.w = (2 / self.n_features) * Q
        return self


class LaplacianFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        ind = np.random.choice(X.shape[0], 2000, replace=False)
        ind1 = ind[:1000]
        ind2 = ind[1000:]
        d = []
        for i in ind1:
            for j in ind2:
                if i != j:
                    distance = np.sum(np.abs(X[i] - X[j]))
                    d.append(distance)
        d = np.array(d)
        sigma = np.median(d)
        gamma = 1.0 / sigma if sigma != 0 else 1.0
        self.w = np.random.standard_cauchy(size=(X.shape[1], self.n_features)) * gamma
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.feature_creator_class = feature_creator_class
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params or {}
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        
        self.func = func
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps = []
        if self.use_PCA:
            pipeline_steps.append(('pca', PCA(n_components=self.new_dim)))
        pipeline_steps.append(('feature_creator', self.feature_creator))
        pipeline_steps.append(('classifier', self.classifier)) 
        self.pipeline = Pipeline(pipeline_steps)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
