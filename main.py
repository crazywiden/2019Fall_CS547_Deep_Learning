import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix
from model import CNN, CNN2
os.chdir(os.getcwd())
logging.basicConfig(filename='record.log', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class cifar10(Dataset):
    def __init__(self, hdf5_path, train=True, transform=None):
        super(cifar10, self).__init__()
        data = h5py.File(hdf5_path, "r")
        if train:
            self.X = data.get("X_train").value
            self.y = data.get("Y_train").value
        else:
            self.X = data.get("X_test").value
            self.y = data.get("Y_test").value
        # need to change data type to unit8 otherwise transform won't work
        if str(self.X.dtype)[:3] == "int":
            self.X = self.X.astype(np.uint8)
            self.y = self.y.astype(np.uint8)
        self.transform = transform

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.X.shape[0]



def parse_args():
    parser = argparse.ArgumentParser(description="hyper-parameters for CNN training on CIFAR10")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="total number of epoches, only integer are allowed")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="batch size for each iteration, only integer are allowed")
    parser.add_argument("--record_step", type=int, default=100,
                        help="how many of iterations take a snapshot of metrics")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate for training")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "RMSprop"],
                        help="choice of optimizer, only Adam and RMSprop are allowed so far")
    parser.add_argument("--model_name", type=str, 
                        help="model will be saved at ./result directory, now give this model a name!")

    args = parser.parse_args()
    return args


def load_data(transform, batch_size):
    transform_train = transform["train_transform"]
    transform_test = transform["test_transform"]

    # train_data = cifar10(data_path, train=True, transform=transform_train)
    # test_data = cifar10(data_path, train=False, transform=transform_test)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def loss_acc_curve(train_loss, train_acc, test_loss, test_acc, plot_path):
    # summarize history for accuracy
    plt.figure(figsize=(20,10))
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "accuracy curve.png"))
    plt.close()
    # summarize history for loss
    plt.figure(figsize=(20,10))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "loss curve.png"))
    plt.close()


def main():

    # set hyper-parameters
    args = parse_args()
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    record_step = args.record_step
    learning_rate = args.lr
    optimizer_name = args.optimizer
    model_name = args.model_name
    assert optimizer_name in ["Adam", "RMSprop"], "optimizer should choose from ['Adam', 'RMSprop']"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.splitext(model_name)[1] != ".ckpt":
        model_name = model_name + ".ckpt"


    root_path = os.getcwd()
    res_path = os.path.join(root_path, "result")
    if not os.path.exists(res_path):
        raise PermissionError("didn't make directory /result!!")

    # load data
    data_transform = {
        "train_transform": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
 
    train_loader, test_loader = load_data(data_transform, batch_size)

    # define the model and optimizer
    model = CNN(num_class=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.1)

    # start training...
    # some variables to record training process
    total_step = len(train_loader)
    start_time = time.perf_counter()
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for n in range(n_epochs):
        print("==============Epoch:{}/{}==============".format(n+1, n_epochs))
        logging.info("==============Epoch:{}/{}==============".format(n+1, n_epochs))
        correct = 0
        num_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.type(torch.long).to(device)

            optimizer.zero_grad()
            # forward
            predicts = model(images)
            loss = criterion(predicts, labels)

            # calculate accuracy
            num_train += labels.size(0)
            _, predict_class = torch.max(predicts.data, 1)
            correct += (predict_class == labels).sum().item()

            # backward and optimize
            
            loss.backward()
            optimizer.step()

            curr_time = time.perf_counter()
            if (i+1) % record_step == 0:
                print("Epoch: {}/{}, step:{}/{}, Loss:{:.4f}, Cumulative accuracy:{:.4f}, time used:{:.4f}".
                    format(n+1, n_epochs, i+1, total_step, loss.item(), correct/num_train, curr_time - start_time))
                logging.info("Epoch: {}/{}, step:{}/{}, Loss:{:.4f}, Cumulative accuracy:{:.4f}, time used:{:.4f}".
                    format(n+1, n_epochs, i+1, total_step, loss.item(), correct/num_train, curr_time - start_time))

        train_loss.append(loss.item())
        train_acc.append(correct/num_train)

        train_accuray = correct/num_train
        scheduler.step()
        # evaluate performance on test data each epoch
        with torch.no_grad():
            correct = 0
            num_test = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.type(torch.long).to(device)

                predicts = model(images)
                loss = criterion(predicts, labels)

                _, predict_class = torch.max(predicts.data, 1)
                num_test += labels.size(0)
                correct += (predict_class == labels).sum().item()

        test_accuracy = correct/num_test
        test_loss.append(loss.item())
        test_acc.append(test_accuracy)

        curr_time = time.perf_counter()
        logging.info("********************************")
        logging.info("Epoch:{}/{}, time used:{:.4f}, train accuracy:{:.4f}, test accuracy:{:.4f}".
            format(n+1, n_epochs, curr_time - start_time, train_accuray, test_accuracy))
        logging.info("********************************")
        print("********************************")
        print("Epoch:{}/{}, time used:{:.4f}, train accuracy:{:.4f}, test accuracy:{:.4f}".
            format(n+1, n_epochs, curr_time - start_time, train_accuray, test_accuracy))
        print("********************************")

    print("model_name:", model_name)
    loss_acc_curve(train_loss, train_acc, test_loss, test_acc, res_path)
    torch.save(model.state_dict(), os.path.join(res_path, model_name))


    # start testing
    model.eval()
    all_predict_class = []
    all_true_class = []
    with torch.no_grad():
        correct = 0
        num_test = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.type(torch.long).to(device)

            predict = model(images)
            _, predict_class = torch.max(predict.data, 1)

            all_true_class.extend(labels.tolist())
            all_predict_class.extend(predict_class.tolist())

            num_test += labels.size(0)
            correct += (labels == predict_class).sum().item()

        test_accuracy = correct/num_test
        print("overall test_accuracy is:{:.4f}".format(test_accuracy))
        logging.info("overall test_accuracy is:{:.4f}".format(test_accuracy))



    
    logging.info("==========test set performance===========")
    report = classification_report(all_true_class, all_predict_class)
    confusion_mat = confusion_matrix(all_true_class, all_predict_class)
    logging.info("{}".format(report))
    logging.info("{}".format(confusion_mat))
    
    print("==========test set performance===========")
    print(classification_report(all_true_class, all_predict_class))
    print(confusion_matrix(all_true_class, all_predict_class))


if __name__ == '__main__':
    main()







            













