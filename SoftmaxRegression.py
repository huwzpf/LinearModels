from Regression import *


class SoftmaxRegression(Regression):
    def __init__(self, x, y, k, norm=Normalization.NO_NORM):
        super().__init__(x.reshape(x.shape[0], -1), y, 0, norm)
        self.k = k
        self.labels = np.zeros((len(y), k))
        for i in range(len(y)):
            self.labels[i, y[i]] = 1
        # initialize Theta as vector of all zeros
        self.Theta = np.zeros((self.args.shape[1], k))

    def print_theta_as_square_matrix(self, n=None, cost=None, t_size=28):
        for i in range(self.Theta.shape[1]):
            w = np.transpose(self.Theta[1:, i]).reshape(t_size, t_size)
            plt.imshow(w, cmap='hot', interpolation='nearest')
            plt.title(f"{n}-th iteration Theta for {i} class with {cost} cost")
            plt.show()

    def validate(self, x, y, norm=Normalization.NO_NORM, err=0.1):
        x_t, y_v = self.prepare_data(norm, x, y)
        if self.polynomial_degree == 0:
            x_v = np.c_[np.ones(x_t.shape[0]), x_t]
        else:
            x_v = generate_polynomial_features(self.polynomial_degree, x_t)

        properly_classified = 0
        failed = 0

        for i in range(len(y_v)):
            k = self.predict(x_v[i, :].reshape(1, -1))
            if np.argmax(k) == y_v[i]:
                properly_classified += 1
            else:
                failed += 1

        return properly_classified, failed

    def plot(self, title):
        self.print_theta_as_square_matrix()

    def clear_theta(self, n):
        self.Theta = np.zeros((n, self.k))

    def test(self, batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost):
        super().test(batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost)

    def train(self, method, n=100, rate=0.1, reg=0):
        super().train(method, n, rate, reg)

    def cost_function(self, reg_term):
        # l(Theta)= sum_i (sum_k ( 1{y_i == k} (h_k) )), where h_k is k-th element of h(X)
        # we can rewrite 1{y_i == k) (h_k) as element wise multiplication of y and h
        # because y is vector of zeros with 1 in i-th place where y_i == k
        rt = math.sqrt(sum(sum(reg_term * np.power(self.Theta, 2))))
        # J(Theta) = m/l(Theta), so it takes positive values and converges to 0
        # (m as a constant in order to avoid rounding errors, without it cost ~= e-5)
        ret = (np.sum((np.sum(np.multiply(self.labels, self.hypothesis(self.args)), axis=1))) + rt)/self.labels.shape[0]
        return 1/ret

    def hypothesis(self, x, i=-1):
        #        [ exp(Theta_1_T * x ]              1
        # h(x) = [       ...         ] * --------------------------
        #        [ exp(Theta_k_T * x ]   sum_j (exp(Theta_j_T * x))
        #                 ^            ^             ^
        #                 h            |             c (constant scalar)
        #                   element-wise multiplication

        # x is m x n and Theta is n x k, so we get m x k vector
        h = np.exp(x.dot(self.Theta))
        c = (1/np.sum(h, axis=1)).reshape(x.shape[0], 1)
        return np.multiply(c, h)


