from Regression import *


class LinearRegression(Regression):
    def __init__(self, x, y, deg=1, norm=Normalization.STD):
        super().__init__(x, y, deg, norm)

    def test(self, batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost):
        super().test(batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, False)

        t1 = time.perf_counter()
        self.normal_eq()
        t2 = time.perf_counter()
        print(f'Cost function after computing normal equation : {self.cost_function(reg_term)} in {t2 - t1} seconds')

        self.plot("NormEq")

        self.clear_theta(self.args.shape[1])

    def train(self, method, n=100, rate=0.1, reg=0):
        super().train(method, n, rate, reg)
        if method == TrainMethod.OTHER:
            self.normal_eq()

    def validate(self, x, y, norm=Normalization.STD, err=0.1):
        x_t, y_v = self.prepare_data(norm, x, y)

        if self.polynomial_degree == 0:
            x_v = np.c_[np.ones(x_t.shape[0]), x_t]
        else:
            x_v = generate_polynomial_features(self.polynomial_degree, x_t)

        properly_classified = 0
        failed = 0

        for i in range(len(y_v)):
            k = self.predict(x_v[i, :].reshape(1, -1))
            if np.abs(k - y_v[i]) < err:
                properly_classified += 1
            else:
                failed += 1

        return properly_classified, failed

    def plot(self, title):
        if self.initial_x.shape[1] == 1:
            plt.scatter(self.initial_x, self.labels)

            # plotting line fit to data
            axes = plt.gca()
            (x_min, x_max) = axes.get_xlim()
            x = np.arange(x_min, x_max, (x_max - x_min) / 100)
            y = generate_polynomial_features(self.polynomial_degree, x.reshape((len(x), 1))).dot(self.Theta)

            plt.title(title)
            plt.plot(x, y)
            plt.show()
        else:
            pass

    def cost_function(self, reg_term):
        # J(Theta) = 1/(2 * n) * sum(h(x)_i - y_i)^2 =
        # = 1/(2 * n) * (hypothesis - y ) * (hypothesis - y)_ T
        h = self.hypothesis(self.args) - self.labels
        return 0.5 * (np.transpose(h).dot(h) + (reg_term * np.transpose(self.Theta).dot(self.Theta))) \
            / self.labels.shape[0]

    def hypothesis(self, x):
        # h(x) = Theta.T * X
        # returning X * Theta
        # n x m ----^    ^---- m x 1, so resulting matrix is n x 1
        # each element of result matrix is Theta.T * X_i
        return x.dot(self.Theta)

    def normal_eq(self):
        # Theta = (X_T * X ) ^-1 * X_T * y
        k = np.transpose(self.args).dot(self.args)
        self.Theta = (np.linalg.inv(k).dot(np.transpose(self.args))).dot(self.labels)




