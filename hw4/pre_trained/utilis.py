import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets


def loss_acc_curve(train_loss, train_acc, test_loss, test_acc, plot_path):
    # summarize history for accuracy
    plt.figure(figsize=(20, 10))
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "accuracy curve.png"))
    plt.close()
    # summarize history for loss
    plt.figure(figsize=(20, 10))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(plot_path, "loss curve.png"))
    plt.close()


def load_CIFAR100(batch_size, transform):
    """
    load data from pytorch
    @parameter:
    root_path -- string
    @returns:
    train_loader, test_loader
    """
    train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True,
                                              transform=transform["train"])
    test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                             transform=transform["test"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_tiny_imageNet(train_path, test_path, batch_size, transform):
    train_set = datasets.ImageFolder(train_path, transform=transform["train"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    if "val_" in os.listdir(test_path):
        create_val_folder(test_path)
    test_set = datasets.ImageFolder(test_path, transform=transform["test"])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def create_val_folder(val_dir):
    path = os.path.join(val_dir, "images")
    filename = os.path.join(val_dir, "val_annotations.txt")

    val_img_dict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]

    for img, label in val_img_dict.items():
        new_path = os.path.join(path, label)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(new_path, img))
