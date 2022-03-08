from Regression import *


class LogisticRegression(Regression):
    def __init__(self, x, y, deg=1, norm=Normalization.STD):
        super().__init__(x, y, deg, norm)

    @staticmethod
    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def test(self, batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost):
        super().test(batch_n, batch_rate, stochastic_n, stochastic_rate, reg_term, log_cost)

        newtons_n = 100
        t1 = time.perf_counter()
        self.newtons_method(newtons_n, reg_term, log_cost)
        t2 = time.perf_counter()
        print(f'Cost function after computing Newton\'s method : {self.cost_function(reg_term)} with'
              f' {reg_term} regularization term in {newtons_n} steps '
              f'and {t2 - t1} seconds')

        self.plot("Newton's method")
        self.clear_theta(self.args.shape[1])

    def train(self, method, n=100, rate=0.1, reg=0):
        super().train(method, n, rate, reg)
        if method == TrainMethod.OTHER:
            self.newtons_method(n, reg)

    def plot(self, title):
        if self.initial_x.shape[1] == 2:
            plt.scatter(self.initial_x[:, 0], self.initial_x[:, 1],
                        c=['#ff2200' if y == 0 else '#1f77b4' for y in self.labels])
            axes = plt.gca()
            (x_min, x_max) = axes.get_xlim()
            (y_min, y_max) = axes.get_ylim()
            # arbitrary number
            elements = self.labels.shape[0] * 2
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, elements), np.linspace(y_min, y_max, elements))
            p = np.empty((elements, elements))
            for i in range(elements):
                for j in range(elements):
                    k = generate_polynomial_features(self.polynomial_degree, np.array([x_grid[i, j], y_grid[i, j]])
                                                     .reshape(1, 2))
                    p[i, j] = self.predict(k)
            plt.contour(x_grid, y_grid, p, levels=[0.5])
            plt.title(title)
            plt.show()
        else:
            pass

    def hypothesis(self, x):
        # h(x) = sigmoid(Theta.T * X)
        # returning X * Theta
        # n x m ----^    ^---- m x 1, so resulting matrix is n x 1
        # each element of result matrix is Theta.T * X_i
        ret = x.dot(self.Theta)
        for i in range(len(ret)):
            ret[i] = self.sigmoid(self, ret[i])
        return ret

    def cost_function(self, reg_term):
        # J(Theta) = 1/m * sum_i(-y_i * log(h(x_i)) - (1-y_i)* log(1-h(x_i))) =
        # -1/m * (y_t * log(h(x)) + (1 - y)_t * log(1 - h(x))) =
        # -1*(transpose(Y) * log(hypothesis()) + transpose(ones(Y.shape) - self.Y) * log(1 - self.hypothesis()))/self.n
        rt = (reg_term * np.transpose(self.Theta).dot(self.Theta))
        return -1*(np.transpose(self.labels).dot(np.log(self.hypothesis(self.args))) +
                   np.transpose(np.ones(self.labels.shape) - self.labels)
                   .dot(np.log(1 - self.hypothesis(self.args))) - rt)/self.labels.shape[0]

    def hessian(self):
        # partial derivative of l(Theta) wrt Theta_j is sum over every training example of (y - h(x)) * x_j
        # partial derivative of above term wrt Theta_j is sum of (h(x) * (1 - h(x)) * x_j * x_i)
        # so Hessian is given by H_i_j = sum_k(h(x_k) * (1 - h(x_k)) * x_i[k] * x_j[k])
        # what can be rewritten as : sum_k(h(x_k) * (1-h(x_k))) * sum_k(x_i[k] * x_j[k])
        # so every element of hessian will be multiplied by constant term sum_k(h(x_k) * (1-h(x_k)))
        # we can build hessian by multiplying matrix H given by H_i_j = sum_k(x_i[k] * x_j[k]) by this constant
        # matrix mentioned above can be calculated as outer product of X
        # so with c = sum(h(x)) * sum(1-h(x)) hessian =  X_T * X * c
        h = self.hypothesis(self.args)
        c = np.sum(h, axis=0) * np.sum((np.ones(h.shape) - h), axis=0)
        return np.transpose(self.args).dot(self.args) * c

    def newtons_method(self, n, reg_term, log_cost=False):
        # update rule :
        # Theta -= Hessian(l(Theta)) ^ -1 * Gradient(l(Theta))
        if log_cost:
            cost_history = np.empty(0)
        for i in range(n):
            if log_cost:
                cost_history = np.append(cost_history, np.array(self.cost_function(reg_term)))
            z = np.linalg.inv(self.hessian()).dot(self.gradient(self.args, self.labels, reg_term))
            self.Theta -= z/np.std(z)
            # division by a constant doesnt change how Theta is fit to data
            # normalize z because it might be too small and cost function tends to
            # not have enough precision and outputs value for Theta = 0
        if log_cost:
            plt.title("cost history")
            plt.scatter(np.arange(0, n), cost_history)
            plt.show()



