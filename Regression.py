import numpy

from Utils import *


class Regression:
    def __init__(self, x, y, deg=1, norm=Normalization.STD):

        self.polynomial_degree = deg
        x_t, y = self.prepare_data(norm, x, y)

        self.initial_x = x_t
        if deg == 0:
            self.args = np.c_[np.ones(x_t.shape[0]), x_t]
        else:
            self.args = generate_polynomial_features(self.polynomial_degree, x_t)
        self.labels = y.reshape((y.size, 1))
        self.Theta = np.zeros((self.args.shape[1], 1))

    def prepare_data(self, norm, x, y):
        if type(x) != numpy.ndarray:
            x = x.to_numpy()
        if type(y) != numpy.ndarray:
            y = y.to_numpy()
        if norm == Normalization.MIN_MAX:
            x_t = self.normalize_min_max(self, x)
        elif norm == Normalization.STD:
            x_t = self.normalize_std(self, x)
        else:
            x_t = x
        return x_t, y

    def validate(self, x, y, norm=Normalization.STD, err=0.1):
        pass



    def train(self, method, n=100, rate=0.1, reg=0):
        if method == TrainMethod.BATCH:
            self.batch_gradient(n, rate, reg, False)
        elif method == TrainMethod.STOCHASTIC:
            self.stochastic_gradient(n, rate, reg, False)

    @staticmethod
    def normalize_std(self, x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    @staticmethod
    def normalize_min_max(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def clear_theta(self, n):
        self.Theta = np.zeros((n, 1))

    def hypothesis(self, x):
        pass

    def cost_function(self, reg_term):
        pass

    def plot(self, title):
        pass

    def predict(self, x):
        return self.hypothesis(x)

    def gradient(self, x, y, reg_term):
        rt = (reg_term * self.Theta)
        rt[0, :] = 0
        return np.transpose(x).dot(self.hypothesis(x) - y) + rt

    def batch_gradient_update(self, learning_rate, reg_term):
        # update rule :
        # Theta_j -= learning_rate * d J(Theta) / d Theta_J , so
        # Theta_j -= learning_rate * (1/n) * (hypothesis(x) - y) * x_j (for single training example)

        # we want to compute for all j (and all training examples) at once,
        # so we create n x m matrix (n - number of training examples, m size of Theta vector)
        # where each element in each row contains update value for each Theta_j
        # and each row is one training example, then we sum all rows and divide by n (take average)
        # and get 1xm row vector containing
        # in j-th column value by which j-th element of Theta needs to be updated

        # multiply each row from h(x) - y by corresponding values in X vector, then sum over all training examples
        # x = (np.sum((self.hypothesis(X) - Y) * X, axis=0) / n )
        # equivalent form
        # np.transpose(x).dot(self.hypothesis(x) - y) / len(y))

        # update Theta
        self.Theta -= learning_rate * self.gradient(self.args, self.labels, reg_term) / self.labels.shape[0]

    def batch_gradient(self, n, learning_rate, reg_term, log_cost):
        if log_cost:
            cost_history = np.empty(0)
        for i in range(n):
            if log_cost:
                cost_history = np.append(cost_history, np.array(self.cost_function(reg_term)))
            self.batch_gradient_update(learning_rate, reg_term)
        if log_cost:
            plt.title("cost history")
            plt.scatter(np.arange(0, n), cost_history)
            plt.show()

    def stochastic_gradient_update(self, learning_rate, reg_term):
        # update rule :
        # for each example :
        # Theta -= learning_rate * 1/n * (h(x_i) - y_i) * x _i
        for i in range(self.labels.shape[0]):
            # reshape because numpy subtraction gets wired
            self.Theta -= (learning_rate * self.gradient(self.args[i, :].reshape(self.args.shape[1], 1).T,
                                                         self.labels[i], reg_term)
                           / self.labels.shape[0]).reshape(self.Theta.shape)

    def stochastic_gradient(self, n, learning_rate, reg_term, log_cost):
        if log_cost:
            cost_history = np.empty(0)
        for i in range(n):
            if log_cost:
                cost_history = np.append(cost_history, np.array(self.cost_function(reg_term)))
            self.stochastic_gradient_update(learning_rate, reg_term)
        if log_cost:
            plt.title("cost history")
            plt.scatter(np.arange(0, n), cost_history)
            plt.show()

    def test(self, batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost):
        self.clear_theta(self.args.shape[1])
        print(f'Zero-initialized Theta cost function : {self.cost_function(reg_term)}')

        if batch_n != 0:
            t1 = time.perf_counter()
            self.batch_gradient(batch_n, batch_rate, reg_term, log_cost)
            t2 = time.perf_counter()
            print(f'Cost function after batch gradient descent : {self.cost_function(reg_term)} in {batch_n} steps'
                  f' with {batch_rate} learning rate and {reg_term} regularization term. Computed in {t2 - t1} seconds')

            self.plot("Batch gradient")
            self.clear_theta(self.args.shape[1])
        if stochastic_n != 0:
            t1 = time.perf_counter()
            self.stochastic_gradient(stochastic_n, stochastic_rate, reg_term, log_cost)
            t2 = time.perf_counter()
            print(
                f'Cost function after stochastic gradient descent : {self.cost_function(reg_term)} in'
                f' {stochastic_n} steps with {stochastic_rate} learning rate and {reg_term} regularization term.'
                f' Computed in {t2 - t1} seconds')

            self.plot("Stochastic gradient")

            self.clear_theta(self.args.shape[1])

