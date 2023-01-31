import numpy as np


class LinearRegression:
    """
    model for the analytical solution
    w = (X^T X)^-1 X^T y
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """fit function for the analytical solution"""

        N, D = X.shape
        ones_to_append = np.ones((N, 1))
        X = np.hstack((ones_to_append, X))

        if np.linalg.det(X.T @ X) != 0:
            temp = np.linalg.inv(X.T @ X) @ X.T @ y
            self.b = temp[0]
            self.w = temp[1:]
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict function for the analytical solution"""
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def calc_MSE(self, w, X, y):
        y_pred = X @ w
        N = len(y)
        loss = np.linalg.norm(y_pred - y) ** 2 * (1 / N)
        return loss

    def calc_grad(self, w, X, y):
        y_pred = X @ w
        N = len(y)

        gradient = np.zeros(w.shape)
        gradient = (-2 / N) * np.matmul(X.T, (y - y_pred))

        return gradient

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.00000025, epochs: int = 100000
    ) -> None:
        """
        fit function for the gradient descent method
        """
        N, D = X.shape

        ones_to_append = np.ones((N, 1))
        X = np.hstack((ones_to_append, X))

        if self.w is None:
            theta_old = np.zeros((D + 1,))
        else:
            theta_old = self.w

        for i in range(epochs):
            loss = self.calc_MSE(theta_old, X, y)
            grad = self.calc_grad(theta_old, X, y)
            theta_new = theta_old - lr * grad
            theta_old = theta_new
            if i % 10000 == 0:
                # print("Current gardient is: " + str(grad))
                print("Current loss is: " + str(loss))

        self.w = theta_new

        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict function for the gradient descent method
        """
        N, D = X.shape
        ones_to_append = np.ones((N, 1))
        X = np.hstack((ones_to_append, X))

        y_pred = X @ self.w
        return y_pred
