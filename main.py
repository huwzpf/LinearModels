from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from SoftmaxRegression import SoftmaxRegression
from Utils import *


if __name__ == '__main__':

    data_linear = pd.read_csv('C:/Users/PCu/Data/data_linear.csv').dropna()
    data_logistic = pd.read_csv('C:/Users/PCu/Data/data_logistic.csv').dropna()

    rLin = LinearRegression(data_linear.iloc[:-10, :-1], data_linear.iloc[:-10, -1], deg=2)
    rLin.test(batch_n=1000, batch_rate=0.01, stochastic_n=1000, stochastic_rate=0.01, reg_term=0, log_cost=False)
    rLin.train(method=TrainMethod.OTHER)
    a, b =rLin.validate(data_linear.iloc[-10:, :-1], data_linear.iloc[-10:, -1])
    print(a, b)
    rLog = LogisticRegression(data_logistic.iloc[:, :-1], data_logistic.iloc[:, -1], deg=2)
    rLog.test(batch_n=10000, batch_rate=0.001, stochastic_n=1000, stochastic_rate=0.001, reg_term=0, log_cost=False)

    train_args = load_training_images('C:/Users/PCu/Data/train-images-idx3-ubyte.gz')
    train_labels = load_training_labels('C:/Users/PCu/Data/train-labels-idx1-ubyte.gz')
    test_args = load_training_images('C:/Users/PCu/Data/t10k-images-idx3-ubyte.gz')
    test_labels = load_training_labels('C:/Users/PCu/Data/t10k-labels-idx1-ubyte.gz')

    rS = SoftmaxRegression(train_args, train_labels, 10)
    rS.test(batch_n=100, batch_rate=0.00001, stochastic_n=100, stochastic_rate=0.00001, reg_term=1, log_cost=True)
    rS.train(TrainMethod.BATCH, n=100, rate=0.0001, reg=1)
    
    good, bad = rS.validate(test_args.reshape(test_args.shape[0], -1), test_labels)
    print(good / (good + bad))





