import os
import time
import h5py
import numpy as np
import warnings
warnings.filterwarnings("error")


def load_data(path):
    """
    load x_train, x_test, y_train, y_test
    from hdf5 file
    @parameter:
    path -- string
    @returns:
    x_train -- np.ndarray
        N by M
        N: number of points
        M: number of features
    x_test -- np.ndarray
        N by M
        N: number of points
        M: number of features
    y_train -- np.ndarray
        N by 1
    y_test -- np.ndarray
        N by 1
    """
    all_data = h5py.File(path, "r+")
    x_train = np.array(all_data["x_train"])
    x_test = np.array(all_data["x_test"])
    y_train = np.array(all_data["y_train"])
    y_test = np.array(all_data["y_test"])
    return x_train, x_test, y_train, y_test

def cal_acc(pred, label):
    """
    definition of accuracy: 
    number of correct prediction/total number of prediction
    @parameter:
    pred -- np.ndarray
        1d vector
    label -- np.ndarray
        1d vector
    @returns:
    acc -- double
    """
    acc = len(np.where(np.equal(pred, label))[0])/len(label)
    return acc

def cal_loss(pred_proba, y):
    """
    definition of loss:
    -\\sum_{y=0}^{K-1}log 1_{y=k} pred_proba_k
    @parameters:
    pred_proba -- np.ndarray
        1d or 2d
        if 2d, pred_proba.shape[0] is number of points
    y -- np.ndarray
        1d vector
        each element is a category
    @returns:
    loss -- double
    """

    loss = 0
    if len(pred_proba.shape) == 1:
        pred_proba = np.expand_dims(pred_proba, axis=0)
    for i in range(pred_proba.shape[0]):
        idx = y[i]
        loss =  loss - np.log(pred_proba[i][idx])
    return loss/pred_proba.shape[0]


def softmax(vec):
    exp_vec = np.exp(vec)
    return exp_vec/sum(exp_vec)

def relu(vec, mode="normal"):
    """
    @parameters:
    vec -- np.ndarray
    mode -- string
        take values of ["normal", "derivative"]
        mode == "normal", calculate relu
        mode == "derivative", calcuate derivative of relu
    @return:
    activ -- np.ndarray
        vector after activation
    """
    if mode == "normal":
        activ = np.zeros(len(vec))
        activ[vec > 0] = np.copy(vec[vec > 0])
    elif mode == "derivative":
        activ = np.zeros(len(vec))
        activ[vec > 0] = 1
    return activ

class one_layer_fc():
    def __init__(self, input_dim, output_dim, hidden_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if "activation" not in kwargs:
            kwargs["activation"] = "relu"
        if kwargs["activation"] == "relu":
            self.activ = relu

        if "init" not in kwargs:
            kwargs["init"] = "Xavier"
            
        if kwargs["init"] == "Gaussian":
            # use N(0, 1) to normalize
            self.W1 = np.random.normal(size=(input_dim, hidden_dim))
            self.b1 = np.random.normal(size=(hidden_dim))
            self.W2 = np.random.normal(size=(hidden_dim, output_dim))
            self.b2 = np.random.normal(size=(output_dim))

        elif kwargs["init"] == "Xavier":
            # use N(0, 1) to normalize
            self.W1 = np.random.normal(loc=0, scale=np.sqrt(2/(input_dim + hidden_dim)), size=(input_dim, hidden_dim))
            self.b1 = np.random.normal(size=(hidden_dim))
            self.W2 = np.random.normal(loc=0, scale=np.sqrt(2/(hidden_dim + output_dim)), size=(hidden_dim, output_dim))
            self.b2 = np.random.normal(size=(output_dim))

    

    def train(self, X, y, lr=0.01):
        for i in range(X.shape[0]): # SGD to train
            self.Z = np.dot(X[i], self.W1) + self.b1
            self.H = self.activ(self.Z)
            self.U = np.dot(self.H, self.W2) + self.b2
            output = softmax(self.U)
            self._update(output, X[i], y[i], lr)

    def _update(self, pred, X, y, lr):
        loss = cal_loss(pred, y)
        dev_U = - (np.eye(len(pred))[y][0] - pred)
        dev_W2 = sum(dev_U) * np.tile(self.H, (self.W2.shape[1], 1))
        dev_W2 = np.transpose(dev_W2)
        self.W2 = self.W2 - lr * dev_W2
        self.b2 = self.b2 - lr * dev_U

        delta = np.dot(self.W2, dev_U)
        dev_Z = np.zeros(len(self.Z))
        dev_Z = self.activ(self.Z, mode="derivative")
        dev_b1 = np.multiply(dev_Z, delta)
        self.W1 = self.W1 - lr * np.outer(X, dev_b1)
        self.b1 = self.b1 - lr * dev_b1


    def predict_proba(self, X):
        all_output = []
        for i in range(len(X)):
            Z = np.dot(X[i], self.W1) + self.b1
            H = self.activ(Z)
            U = np.dot(H, self.W2) + self.b2
            output = softmax(U)
            all_output.append(output)
        return np.array(all_output)
        

    def predict_classes(self, X):
        output = self.predict_proba(X)
        return np.argmax(output, axis=1)



if __name__ == '__main__':
    root_path = r"D:\学习\Graduate\IE534_deep_learning\hw1"
    file_path = "MNISTdata.hdf5"
    
    x_train, x_test, y_train, y_test = load_data(os.path.join(root_path, file_path))

    # hyper parameters
    n_epochs = 7
    learning_rate = 0.01
    hidden_unit = int(256)
    activation = "relu"

    input_dim = x_train.shape[1]
    output_dim = len(np.unique(y_train))


    # training 
    model = one_layer_fc(input_dim, output_dim, hidden_dim=hidden_unit)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    start_t = time.perf_counter()
    for n in range(n_epochs):
        epoch_start = time.perf_counter()
        # each epoch re-shuffle the data
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        model.train(x_train, y_train)

        output_train = model.predict_proba(x_train)
        output_test = model.predict_proba(x_test)

        y_pred_train = model.predict_classes(x_train)
        y_pred_test = model.predict_classes(x_test)

        train_acc.append(cal_acc(y_pred_train, y_train.T[0]))
        test_acc.append(cal_acc(y_pred_test, y_test.T[0]))

        train_loss.append(cal_loss(output_train, y_train))
        test_loss.append(cal_loss(output_test, y_test))

        end_time = time.perf_counter()
        print("current epoch : %s/%s, epoch time: %s, total time: %s" % 
            (n, n_epochs, str(end_time - epoch_start), str(end_time - start_t)))
        print("train accuracy: %s, test accuracy: %s " % (str(train_acc[-1]), str(test_acc[-1])))






