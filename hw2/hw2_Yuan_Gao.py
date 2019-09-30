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


def relu(tensor, mode="normal"):
    """
    @parameters:
    tensor -- np.ndarray
    mode -- string
        take values of ["normal", "derivative"]
        mode == "normal", calculate relu
        mode == "derivative", calcuate derivative of relu
    @return:
    activ -- np.ndarray
        vector after activation
    """
    if mode == "normal":
        tensor[tensor <= 0] = 0
    elif mode == "derivative":
        tensor[tensor > 0] = 1
        tensor[tensor <= 0] = 0
    return tensor


def convolve(mat, filter_mat):
    if len(filter_mat.shape) == 2:
        filter_mat = filter_mat[np.newaxis, :, :]

    num_channel = filter_mat.shape[0]
    m = filter_mat.shape[1]
    n = filter_mat.shape[2]

    y_size = mat.shape[0] - m + 1
    x_size = mat.shape[1] - n + 1

    Z = np.zeros((num_channel, y_size, x_size))
    for c in range(num_channel):
        for i in range(y_size):
            for j in range(x_size):
                Z[c][i][j] = np.sum(mat[i:i+m, j:j+n] * filter_mat[c, :, :])
    return Z


class one_layer_CNN():
    def __init__(self, input_size, output_dim, filter_size, **kwargs):
        self.input_size = input_size
        self.output_dim = output_dim
        self.filter_size = filter_size


        if "activation" not in kwargs:
            kwargs["activation"] = "relu"
        if kwargs["activation"] == "relu":
            self.activ = relu

        if "init" not in kwargs:
            kwargs["init"] = "Gaussian"
        
        if len(filter_size) == 2: # only one channel
            filter_size = (1, ) + filter_size

        num_channel = filter_size[0]
        hidden_size1 = input_size[0] - filter_size[1] + 1
        hidden_size2 = input_size[1] - filter_size[2] + 1
        
        if kwargs["init"] == "Gaussian":
            # use N(0, 1) to normalize
            self.K = np.random.normal(size=filter_size)
            self.W = np.random.normal(size=(output_dim, num_channel, hidden_size1, hidden_size2))
            self.b = np.random.normal(size=(output_dim))

        elif kwargs["init"] == "Xavier":
            # use N(0, 1) to normalize
            self.K = np.random.normal(loc=0, scale=np.sqrt(1/(filter_size[0]*filter_size[1])), size=filter_size)
            self.W = np.random.normal(loc=0, scale=np.sqrt(1/(output_dim*hidden_size1*hidden_size2)), 
                size=(output_dim, num_channel, hidden_size1, hidden_size2))
            self.b = np.random.normal(size=(output_dim))


    def train(self, X, y, lr=0.01):
        for i in range(X.shape[0]): # SGD to train
            sample_X = X[i, :, :]    
            output = self.forward(sample_X)
            self.backward(output, sample_X, y[i], lr)


    def forward(self, sample_X):
        self.Z = convolve(sample_X, self.K)
        self.H = self.activ(self.Z)
        self.U = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            self.U[i] = np.sum(np.multiply(self.W[i, :, :, :], self.H)) + self.b[i]
        output = softmax(self.U)
        return output


    def backward(self, pred, X, y, lr):
        dev_U = - (np.eye(len(pred))[y][0] - pred)

        dev_W = np.zeros((output_dim, self.Z.shape[0], self.Z.shape[1], self.Z.shape[2]))
        for i in range(output_dim):
            dev_W[i, :, :, :] = dev_U[i] * self.H

        self.W = self.W - lr * dev_W
        self.b = self.b - lr * dev_U

        delta = np.zeros_like(self.Z)

        # this part can be optimized
        for c in range(self.Z.shape[0]):
            for i in range(self.Z.shape[1]):
                for j in range(self.Z.shape[2]):
                    delta[c][i][j] = np.dot(dev_U, self.W[:, c, i, j])


        dev_Z = self.activ(self.Z, mode="derivative")
        dev_K = convolve(X, np.multiply(dev_Z, delta))
        self.K = self.K - lr * dev_K



    def predict_proba(self, X):
        all_output = []
        for i in range(X.shape[0]):
            output = self.forward(X[i, :, :])
            all_output.append(output)
        return np.array(all_output)
        

    def predict_classes(self, X):
        output = self.predict_proba(X)
        return np.argmax(output, axis=1)



if __name__ == '__main__':
    root_path = r"/Users/widen/Documents/courses/IE534_deep_learning/hw2/"
    file_path = "MNISTdata.hdf5"
    
    x_train, x_test, y_train, y_test = load_data(os.path.join(root_path, file_path))

    # hyper parameters
    n_epochs = 10
    learning_rate = 0.01
    activation = "relu"

    output_dim = len(np.unique(y_train))

    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    # training 
    model = one_layer_CNN(input_size=x_train[0].shape, output_dim=output_dim, filter_size=(5, 5))

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






