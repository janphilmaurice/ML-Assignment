import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1 - sig)



def one_hot1(y):
    labels = np.unique(y)
    labels_dic = {}
    for i in range(len(labels)):
        labels_dic[labels[i]] = i
    one_hot_y = np.zeros((len(y), len(labels)))
    #print(y.index[-1])
    for idx,i in enumerate(y):
        if i < len(y):
            one_hot_y[i, labels_dic[y[i]]] = 1
        #print('yi:',i)
        #one_hot_y[idx, labels_dic[y[i]]] = 1

    return labels, one_hot_y

def one_hot(y):
    labels = np.unique(y)
    labels_dic = {label: i for i, label in enumerate(labels)}
    one_hot_y = np.eye(len(labels))[np.vectorize(labels_dic.get)(y)]
    return labels, one_hot_y


def normalisation(x, min_x=None, diff=None):
    if min_x is None or diff is None:
        max_x = np.max(x, axis=0)
        str_values = [x for x in max_x if isinstance(x, str)]

        min_x = np.min(x, axis=0)
        str_values = [x for x in min_x if isinstance(x, str)]

        diff = max_x - min_x
    diff[diff == 0] = 1
    return (x - min_x) / diff, min_x, diff


def anti_normalisation(x, min_x, diff):
    return x * diff + min_x


def binary_cross_entropy(y_true, y_pred):
    # Make sure the predicted values are between (0, 1) to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculating cross entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss

class Layer(object):
    def __init__(self, NI: int, NO: int):
        self.NI = NI
        self.NO = NO

        self.w = None  # weight
        self.b = None  # bias
        self.dw = None  # delta weight
        self.db = None  # delta bias

        self.x = None  # input
        self.z = None  # linear result before activation
        self.o = None  # output

        self.randomise()

    def randomise(self):
        self.w = np.random.randn(self.NI, self.NO)
        self.b = np.zeros(self.NO)

    def forward(self):
        self.z = np.dot(self.x, self.w) + self.b
        self.o = sigmoid(self.z)
        return self.o

    def update(self):
        self.w += self.dw
        self.b += self.db


class Optimizer(object):
    def __init__(self):
        self.NI = None
        self.NO = None
        self.NHs = None
        self.NHL = None

    def construct(self, NI: int, NO: int, NHs: list):
        self.NI = NI
        self.NO = NO
        self.NHs = NHs
        self.NHL = len(NHs)
        self.initialise()

    def initialise(self):
        pass

    def optimize(self, i, gw, gb, epoch) -> tuple[float, float]:
        pass


class Momentum(Optimizer):
    def __init__(self, mf=0.9):
        super().__init__()
        self.vws = None  # speeds of weights
        self.vbs = None  # speeds of biases
        self.mf = mf  # momentum factor

    def initialise(self):
        self.vws = []
        self.vbs = []
        self.vws.append(np.zeros([self.NI, self.NHs[0]]))
        for i in range(self.NHL - 1):
            self.vws.append(np.zeros([self.NHs[i], self.NHs[i + 1]]))
        self.vws.append(np.zeros([self.NHs[-1], self.NO]))
        for i in range(self.NHL):
            self.vbs.append(np.zeros(self.NHs[i]))
        self.vbs.append(np.zeros(self.NO))

    def optimize(self, i, gw, gb, epoch):
        self.vws[i] = self.mf * self.vws[i] + (1 - self.mf) * gw
        self.vbs[i] = self.mf * self.vbs[i] + (1 - self.mf) * gb
        dw = self.vws[i]
        db = self.vbs[i]
        return dw, db


class RMSProp(Optimizer):
    def __init__(self, df=0.999):
        super().__init__()
        self.lws = None  # last weights
        self.lbs = None  # last biases
        self.df = df  # decay factor

    def initialise(self):
        self.lws = []
        self.lbs = []
        self.lws.append(np.zeros([self.NI, self.NHs[0]]))
        for i in range(self.NHL - 1):
            self.lws.append(np.zeros([self.NHs[i], self.NHs[i + 1]]))
        self.lws.append(np.zeros([self.NHs[-1], self.NO]))
        for i in range(self.NHL):
            self.lbs.append(np.zeros(self.NHs[i]))
        self.lbs.append(np.zeros(self.NO))

    def optimize(self, i, gw, gb, epoch):
        e = 10 ** (-6)
        self.lws[i] = self.df * self.lws[i] + (1 - self.df) * (gw * gw)
        self.lbs[i] = self.df * self.lbs[i] + (1 - self.df) * (gb * gb)
        dw = 1 / (np.sqrt(self.lws[i] + e)) * gw
        db = 1 / (np.sqrt(self.lbs[i] + e)) * gb
        return dw, db


class Adam(Optimizer):
    def __init__(self, mf=0.9, df=0.999):
        super().__init__()
        self.vws = None  # speeds of weights
        self.vbs = None  # speeds of biases
        self.mf = mf  # momentum factor
        self.lws = None  # last weights
        self.lbs = None  # last biases
        self.df = df  # decay factor

    def initialise(self):
        self.vws = []
        self.vbs = []
        self.vws.append(np.zeros([self.NI, self.NHs[0]]))
        for i in range(self.NHL - 1):
            self.vws.append(np.zeros([self.NHs[i], self.NHs[i + 1]]))
        self.vws.append(np.zeros([self.NHs[-1], self.NO]))
        for i in range(self.NHL):
            self.vbs.append(np.zeros(self.NHs[i]))
        self.vbs.append(np.zeros(self.NO))
        self.lws = []
        self.lbs = []
        self.lws.append(np.zeros([self.NI, self.NHs[0]]))
        for i in range(self.NHL - 1):
            self.lws.append(np.zeros([self.NHs[i], self.NHs[i + 1]]))
        self.lws.append(np.zeros([self.NHs[-1], self.NO]))
        for i in range(self.NHL):
            self.lbs.append(np.zeros(self.NHs[i]))
        self.lbs.append(np.zeros(self.NO))

    def optimize(self, i, gw, gb, epoch):
        e = 10 ** (-8)

        self.vws[i] = self.mf * self.vws[i] + (1 - self.mf) * gw
        self.vbs[i] = self.mf * self.vbs[i] + (1 - self.mf) * gb
        self.lws[i] = self.df * self.lws[i] + (1 - self.df) * (gw * gw)
        self.lbs[i] = self.df * self.lbs[i] + (1 - self.df) * (gb * gb)

        # eliminate deviation
        vw_hat = self.vws[i] / (1 - np.power(self.mf, epoch))
        vb_hat = self.vbs[i] / (1 - np.power(self.mf, epoch))
        lw_hat = self.lws[i] / (1 - np.power(self.df, epoch))
        lb_hat = self.lbs[i] / (1 - np.power(self.df, epoch))

        dw = 1 / (np.sqrt(lw_hat) + e) * vw_hat
        db = 1 / (np.sqrt(lb_hat) + e) * vb_hat
        return dw, db


class MLP(object):
    def __init__(self, NI: int, NO: int, NHs: list, lr=0.001, optimizer: Optimizer = None):
        # structure attributes
        self.NI = NI  # number of inputs
        self.NO = NO  # number of outputs
        self.NHs = NHs  # number of hidden units in hidden layers
        self.NHL = len(NHs)  # number of hidden layers
        self.lr = lr  # learning rate
        self.optimizer = optimizer  # optimizer: momentum, RMSProp or Adam
        self.layers = None  # hidden layers
        self.h = None  # cache of delta

        # attributes about data
        self.X = None
        self.y = None
        self.min_X = None
        self.diff_X = None
        self.min_y = None
        self.diff_y = None
        self.labels = None  # all classes

        self.errs = None  # all errors
        self.is_print = True  # whether print error and accuracy

        self.initialise()

    def initialise(self):
        self.layers = []
        self.layers.append(Layer(self.NI, self.NHs[0]))
        for i in range(self.NHL - 1):
            self.layers.append(Layer(self.NHs[i], self.NHs[i + 1]))
        self.layers.append(Layer(self.NHs[-1], self.NO))
        self.errs = []
        if self.optimizer is not None:
            self.optimizer.construct(NI=self.NI, NO=self.NO, NHs=self.NHs)

    def forward(self, X):
        self.layers[0].x = X
        for i in range(len(self.layers) - 1):
            self.layers[i].forward()
            self.layers[i + 1].x = self.layers[i].o
        self.layers[-1].forward()

    def backward(self, y, epoch):
        if self.optimizer is None:
            # the last hidden layer - output, f'(x) = sigmoid_der
            self.h = (self.layers[-1].o - y) * sigmoid_der(self.layers[-1].z)
            self.layers[-1].dw = -self.lr * np.dot(np.transpose(self.layers[-1].x), self.h)
            self.layers[-1].db = -self.lr * np.sum(self.h, axis=0)

            # others, f'(x) = sigmoid_der
            for i in range(self.NHL):
                o1 = self.NHL - i  # o+1
                o = o1 - 1
                self.h = self.h.dot(np.transpose(self.layers[o1].w)) * sigmoid_der(self.layers[o].z)
                self.layers[o].dw = -self.lr * np.dot(np.transpose(self.layers[o].x), self.h)
                self.layers[o].db = -self.lr * np.sum(self.h, axis=0)
        else:
            # the last hidden layer - output, f'(x) = sigmoid_der
            self.h = (self.layers[-1].o - y) * sigmoid_der(self.layers[-1].z)
            gw = np.dot(np.transpose(self.layers[-1].x), self.h)
            gb = np.sum(self.h, axis=0)
            dw, db = self.optimizer.optimize(-1, gw, gb, epoch)
            self.layers[-1].dw, self.layers[-1].db = -self.lr * dw, -self.lr * db

            # others, f'(x) = sigmoid_der
            for i in range(self.NHL):
                o1 = self.NHL - i  # o+1
                o = o1 - 1
                self.h = self.h.dot(np.transpose(self.layers[o1].w)) * sigmoid_der(self.layers[o].z)
                gw = np.dot(np.transpose(self.layers[o].x), self.h)
                gb = np.sum(self.h, axis=0)
                dw, db = self.optimizer.optimize(o, gw, gb, epoch)
                self.layers[o].dw, self.layers[o].db = -self.lr * dw, -self.lr * db

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, epoch, batch_size):
        batch_num = int(len(self.X) / batch_size)
        for i in range(1, epoch + 1):
            #print('epoch ===>', i)
            err = 0
            for j in range(batch_num):
                start = j * batch_size
                end = (j + 1) * batch_size
                train_y = self.y[start:end]
                self.forward(self.X[start:end])
                self.backward(train_y, i)
                self.update()
                #err += np.sum((self.layers[-1].o - train_y) ** 2 / 2)
                err += binary_cross_entropy(y_true=train_y, y_pred=self.layers[-1].o)
            self.errs.append(err / batch_size / batch_num)  # 均方误差
            if self.is_print:
                if i % 100 == 0:
                    #print("Error at epoch {} is {}.".format(i, self.errs[i - 1]))
                    pass
                    if self.NO > 1:
                        #print("Accuracy at epoch {} is {}.".format(i, self.compute_accuracy()))
                        pass

    def fit(self, X, y, epoch: int, batch_size: int = 1):
        self.X, self.min_X, self.diff_X = normalisation(X)
        if self.NO > 1:     # number of outputs
            self.labels, self.y = one_hot(y)
        else:
            self.y = np.reshape(y, (y.shape[0], 1))
            self.y, self.min_y, self.diff_y = normalisation(self.y)
        self.train(epoch, batch_size)

    def predict(self, X, normalised=False):
        if not normalised:
            X = normalisation(X, self.min_X, self.diff_X)[0]
        else:
            X = X
        self.forward(X)
        o = self.layers[-1].o
        if self.NO == 1:
            return anti_normalisation(o, self.min_y, self.diff_y)
        else:
            y = [self.labels[i] for i in np.argmax(o, axis=1)]
            return y

    def write_train_data(self, filename):
        with open(filename, "w") as file:
            for i in range(len(self.errs)):
                file.write("Error at epoch {} is {}.\n".format(i + 1, self.errs[i]))

    def show_error_plot(self):
        plt.plot([i for i in range(len(self.errs))], self.errs)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

    def compute_accuracy(self):
        y_pre = self.predict(self.X, True)
        y_true = [self.labels[i] for i in np.argmax(self.y, axis=1)]
        accuracy = 0
        for i in range(len(y_pre)):
            if y_pre[i] == y_true[i]:
                accuracy += 1
        accuracy = accuracy / len(y_pre)
        return accuracy
