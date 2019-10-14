import os
import time
import logging
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
# from resnet_model import resnet34
from model import ResNet, Block
from utilis import load_tiny_imageNet, loss_acc_curve
os.chdir(os.getcwd())
logging.basicConfig(filename='tiny_ImageNet_record.log', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
CUDA = torch.cuda.is_available()


def arg_parser():
    parser = argparse.ArgumentParser(description="hyper-parameters for CNN training on CIFAR10")
    parser.add_argument("--image_size", type=int, default=32,
                        help="side length of the input images, suppose all images are square")
    parser.add_argument("--num_class", type=int, default=100,
                        help="number of class")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="total number of epoches, only integer are allowed")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for each iteration, only integer are allowed")
    parser.add_argument("--record_step", type=int, default=100,
                        help="how many of iterations take a snapshot of metrics")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for training")
    parser.add_argument("--optimizer", type=str, default="RMSprop", choices=["Adam", "RMSprop"],
                        help="choice of optimizer, only Adam and RMSprop are allowed so far")
    parser.add_argument("--model_name", type=str, default="",
                        help="model will be saved at ./result directory")

    args = parser.parse_args()
    return args


def train(train_loader, test_loader, model, loss_func, optimizer, n_epochs, record_step=100):
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    total_step = len(train_loader)
    start_time = time.perf_counter()
    for n in range(n_epochs):
        print("================Epoch:{}/{}====================".format(n + 1, n_epochs))
        correct = 0
        num_sample = 0

        for i, (images, labels) in enumerate(train_loader):
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()

            # forward
            predicts = model(images)
            loss = loss_func(predicts, labels)

            # calculate accuracy
            num_sample += labels.size(0)
            _, predict_class = torch.max(predicts.data, 1)
            correct += (predict_class == labels).sum().item()

            loss.backward()
            optimizer.step()

            curr_time = time.perf_counter()
            if (i + 1) % record_step == 0:
                print("Epoch: {}/{}, step:{}/{}, Loss:{:.4f}, Cumulative accuracy:{:.4f}, time used:{:.4f}".
                      format(n + 1, n_epochs, i + 1, total_step, loss.item(), correct / num_sample,
                             curr_time - start_time))
                logging.info("Epoch: {}/{}, step:{}/{}, Loss:{:.4f}, Cumulative accuracy:{:.4f}, time used:{:.4f}".
                             format(n + 1, n_epochs, i + 1, total_step, loss.item(), correct / num_sample,
                                    curr_time - start_time))

        train_loss.append(loss.item())
        train_accuracy = correct / num_sample
        train_acc.append(train_accuracy)

        with torch.no_grad():
            correct = 0
            num_sample = 0
            for images, labels in test_loader:
                if CUDA:
                    images = images.cuda()
                    labels = labels.cuda()

                predicts = model(images)
                loss = loss_func(predicts, labels)

                _, predict_class = torch.max(predicts, 1)
                correct += (predict_class == labels).sum().item()
                num_sample += labels.size(0)
            test_accuracy = correct / num_sample
            test_acc.append(test_accuracy)
            test_loss.append(loss.item())
        test_accuracy = correct / num_sample
        test_loss.append(loss.item())
        test_acc.append(test_accuracy)

        curr_time = time.perf_counter()
        logging.info("********************************")
        logging.info("Epoch:{}/{}, time used:{:.4f}, train accuracy:{:.4f}, test accuracy:{:.4f}".
                     format(n + 1, n_epochs, curr_time - start_time, train_accuracy, test_accuracy))
        logging.info("********************************")
        print("********************************")
        print("Epoch:{}/{}, time used:{:.4f}, train accuracy:{:.4f}, test accuracy:{:.4f}".
              format(n + 1, n_epochs, curr_time - start_time, train_accuracy, test_accuracy))
        print("********************************")

    metrics = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc
    }
    return model, metrics


def prediction(model, test_loader):
    model.eval()
    all_predict_class = []
    all_true_class = []
    with torch.no_grad():
        correct = 0
        num_test = 0
        for images, labels in test_loader:
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()

            predict = model(images)
            _, predict_class = torch.max(predict.data, 1)

            all_true_class.extend(labels.tolist())
            all_predict_class.extend(predict_class.tolist())

            num_test += labels.size(0)
            correct += (labels == predict_class).sum().item()

        test_accuracy = correct / num_test
        print("overall test_accuracy is:{:.4f}".format(test_accuracy))
        logging.info("overall test_accuracy is:{:.4f}".format(test_accuracy))
    return all_true_class, all_predict_class


def main():
    # set hyper-parameters
    args = arg_parser()
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    record_step = args.record_step
    learning_rate = args.lr
    optimizer_name = args.optimizer
    image_size = args.image_size
    num_class = args.num_class
    model_name = args.model_name

    assert optimizer_name in ["Adam", "RMSprop", "SGD"], "optimizer should choose from ['Adam', 'RMSprop']"
    assert len(model_name) > 0, "you should give the model a name to save/load"

    if os.path.splitext(model_name)[1] != ".ckpt":
        model_name = model_name + ".ckpt"

    root_path = os.getcwd()
    res_path = os.path.join(root_path, "result")

    if not os.path.exists(res_path):
        raise PermissionError("didn't make directory /result!!")

    # load data
    train_mean = [0.4914, 0.4822, 0.4465]
    train_std = [0.2023, 0.1994, 0.2010]
    transform = {
        "train": transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
    }

    # train_path = r"/u/training/tra330/scratch/hw4/data/tiny-imagenet-200/train"
    # test_path = r"/u/training/tra330/scratch/hw4/data/tiny-imagenet-200/val"

    train_path = r"/Users/widen/Documents/study/IE534_deep_learning/hw4/tiny-imagenet-200/train"
    test_path = r"/Users/widen/Documents/study/IE534_deep_learning/hw4/tiny-imagenet-200/val"
    train_loader, test_loader = load_tiny_imageNet(train_path, test_path, batch_size, transform)
    # define the model and optimizer
    # model = resnet34()
    model = ResNet(Block, [2, 4, 4, 2], num_class)
    if CUDA:
        model = model.cuda()
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # start training...
    model, metrics = train(train_loader, test_loader,
                           model, criterion, optimizer,
                           n_epochs, record_step=record_step)

    train_loss, test_loss = metrics["train_loss"], metrics["test_loss"]
    train_acc, test_acc = metrics["train_acc"], metrics["test_acc"]
    loss_acc_curve(train_loss, train_acc, test_loss, test_acc, res_path)
    torch.save(model.state_dict(), os.path.join(res_path, model_name))

    # start testing
    heuristic_true_class, heuristic_predict_class = prediction(model, test_loader)

    report = classification_report(heuristic_true_class, heuristic_predict_class)
    confusion_mat = confusion_matrix(heuristic_true_class, heuristic_predict_class)
    logging.info("==========heuristic test set performance===========")
    logging.info("{}".format(report))
    logging.info("{}".format(confusion_mat))

    print("==========heuristic  test set performance===========")
    print(report)
    print(confusion_mat)


if __name__ == '__main__':
    main()