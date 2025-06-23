import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :parame kernel_scale: length scale in RBF kernel formula
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        K = self.kernel(x, x)
        return 0.5 * np.linalg.norm(K @ self.w - y)**2 + self.regularization * 0.5 * self.w.T @ (K @ self.w)

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        K = self.kernel(x, x)
        return K @ (K @ self.w - y) + self.regularization * (K @ self.w)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """
        K = self.kernel(x, x)
        self.x = x
        self.w = np.zeros(K.shape[0])
        prev_w = np.inf * np.ones(K.shape[0])
        i = 0
        while (i < self.max_iter and np.linalg.norm(self.w - prev_w) ** 2 > self.tolerance):
            prev_w = self.w.copy()
            grad = self.calc_grad(x, y)
            self.w -= self.lr * grad
            self.loss_history.append(self.calc_loss(x, y))
            i += 1
        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """
        K = self.kernel(x, x)
        self.x = x
        A = self.K + self.regularization * np.eye(K.shape[0])
        self.w = np.linalg.solve(A, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.kernel(x, self.x) @ self.w