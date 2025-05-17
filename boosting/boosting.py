from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = None,
        subsample: int | float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = None,
        rsm: int | float = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255,
        goss: bool | None = None,
        goss_k: float | int = 0.2,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        
        self.early_stopping_rounds: int | None = early_stopping_rounds

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * (1 - self.sigmoid(y * z))
        
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type.lower() if bootstrap_type is not None else None
        
        self.rsm = rsm
        self.quantization_type = quantization_type.lower() if quantization_type is not None else None
        self.nbins = nbins
        
        self.goss = goss
        self.goss_k = goss_k
        
    def _get_bootstrapped_sample(self, X, y, gradients):
        if self.bootstrap_type == 'bernoulli':
            sample_indices = np.random.choice(np.arange(X.shape[0]), size=int(self.subsample * X.shape[0]), replace=True)
            return sample_indices
        elif self.bootstrap_type == 'bayesian':
            weights = (-np.log(np.random.uniform(size=X.shape[0]))) ** self.bagging_temperature
            sample_indices = np.random.choice(np.arange(X.shape[0]), p=weights/(weights.sum()), size=int(self.subsample * X.shape[0]), replace=True)
            return sample_indices
        elif self.bootstrap_type == 'goss':
            sorted_indices = np.argsort(np.abs(gradients))[::-1]
            large_gradient_count = max(int(len(sorted_indices) * self.goss_k), 1)
            large_gradient_indices = sorted_indices[:large_gradient_count]
            small_gradient_indices = sorted_indices[large_gradient_count:]
            small_gradient_subsample_size = max(int(len(small_gradient_indices) * self.subsample), 1)
            small_gradient_sampled_indices = np.random.choice(small_gradient_indices, size=small_gradient_subsample_size, replace=False)
            sampled_indices = np.concatenate([large_gradient_indices, small_gradient_sampled_indices])
            return sampled_indices
    
    def feature_importances_(self):
        importance_sum = 0
        importance_dict = {}
        for model in self.models:
            importance = model.feature_importances_
            importance_sum += sum(importance)
            importance_dict.update({i: importance_sum for i, importance_sum in enumerate(importance)})
        return importance_dict

    def partial_fit(self, X, y, pred):
        x_bootstrap = X
        y_bootstrap = y
        ind = np.arange(X.shape[0])
        if self.bootstrap_type is not None or self.goss:
            gradients = self.loss_derivative(y, pred)
            ind = self._get_bootstrapped_sample(X, y, gradients)
            x_bootstrap = X[ind]
            y_bootstrap = y[ind]
        if self.rsm != 1.0:
            x_bootstrap = self._get_random_features(X)
        if self.quantization_type is not None:
            x_bootstrap = self._quantize_data(X)
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_bootstrap, self.loss_derivative(y_bootstrap, pred[ind]))
        predictions = model.predict(X)
        gamma = self.find_optimal_gamma(y, pred, predictions)
        pred += self.learning_rate * gamma * predictions
        self.models.append(model)
        self.gammas.append(gamma)
        # raise Exception("partial_fit method not implemented")

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])
        if y_val is not None:
            val_predictions = np.zeros(y_val.shape[0])

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)
            self.history['train_roc_auc'].append(roc_auc_score(y_train, self.sigmoid(train_predictions)))
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            if X_val is not None and y_val is not None:
                val_pred_update = self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)
                val_predictions += val_pred_update
                val_loss = self.loss_fn(y_val, val_predictions)
                self.history['val_roc_auc'].append(roc_auc_score(y_val, self.sigmoid(val_predictions)))
                self.history['val_loss'].append(val_loss)
                if self.early_stopping_rounds is not None:
                    if i >= self.early_stopping_rounds and val_loss >= min(self.history['val_loss'][-self.early_stopping_rounds:]):
                        break
                    self.history['val_loss'][i % self.early_stopping_rounds] = val_loss

        if plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], 2))
        for model, gamma in zip(self.models, self.gammas):
            predictions[:, 1] += self.learning_rate * gamma * model.predict(X)
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions
        # raise Exception("predict_proba method not implemented")

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_roc_auc'], label='train')
        if 'val_roc_auc' in self.history:
            plt.plot(self.history['val_roc_auc'], label='val')
        plt.ylim((0, 1.1))
        plt.xlabel('Estimators')
        plt.ylabel('ROC-AUC')
        plt.title('ROC-AUC')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='train')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='val')
        plt.xlabel('Estimators')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.show()
        # raise Exception("plot_history method not implemented")
