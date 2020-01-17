import numpy as np
import cvxopt
import copy

class Perceptron():
    def __init__(self, lr=0.01, num_epochs=50, seed=2020):
        self.lr = lr
        self.num_epochs = num_epochs
        self.seed = seed
        self.err = []

    def fit(self, X, y):
        #for reproducing the same results
        random_generator = np.random.RandomState(self.seed)
        # X = np.insert(X, 0, 1, axis=1)
        x_rows, x_columns = X.shape
        self.weights = random_generator.normal(loc=0.0, scale=0.001,
                                                size=x_columns + 1)
        self.b = self.weights[0]
        self.weights = self.weights[1:]

        for e in range(self.num_epochs):
            errors = 0
            for sample, gt in zip(X, y):
                prediction = self.predict(sample)
                delta = self.lr * (gt - prediction)
                self.weights += delta * sample
                self.b += delta
                errors += int(delta != 0.0)
            self.err.append(errors)
        return

    def predict(self, X):
        z = np.dot(X, self.weights) + self.b
        return np.sign(z)


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T).astype(float)


def polynomial_kernel(x1, x2, p=3):
    return (1 + np.dot(x1, x2.T)) ** p


class SVM():
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = float(C) if C is not None else C

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # gram matrix
        K = self.kernel(X, X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp = np.ones(n_samples)
            G = cvxopt.matrix(np.vstack((np.diag(-1.0 * tmp),
                                            np.identity(n_samples))))
            h = cvxopt.matrix(np.hstack((0.0 * tmp, self.C * tmp)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # lagrange multipliers
        a = np.ravel(solution['x'])

        # support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        self.a = a * sv
        self.sv = np.zeros(X.shape)
        self.sv[sv] = X[sv]
        self.sv_y = np.zeros(y.shape)
        self.sv_y[sv] = y[sv]
        self.b = (np.sum(self.sv_y) - np.sum(self.a * self.sv_y * K[sv, :])) / sum(sv)
        # weight vector
        if self.kernel == linear_kernel:
            self.w = np.dot(self.a * self.sv_y, self.sv)
        else:
            self.w = None

        print("%d support vectors out of %d points" % (sum(sv), n_samples))

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.sum(self.a * self.sv_y * self.kernel(X, self.sv), axis=1)
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
