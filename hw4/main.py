import os
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.distributed as dist
from mpi4py import MPI
from utilis import load_CIFAR100
from model import Net2

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
    ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)


# load CIFAR100 data
train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]
transform = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
    }
train_loader, test_loader = load_CIFAR100(batch_size=64, transform=transform)


model = Net2()

# Make sure that all nodes have the same model
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0 / np.float(num_nodes)

model.cuda()

path_save = os.path.join(os.getcwd(), "result")

LR = 0.001
batch_size = 100
Num_Epochs = 1000

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


start_time = time.perf_counter()
for epoch in range(Num_Epochs):
    print("================Epoch:{}/{}====================".format(epoch + 1, Num_Epochs))
    model.train()
    for i, (images, labels) in train_loader:
        # data, target = Variable(x_train_batch), Variable(y_train_batch)
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # calc gradients

        for param in model.parameters():
            # print(param.grad.data)
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()

        optimizer.step()  # update gradients

    model.eval()
    # Train Loss
    counter = 0
    train_accuracy_sum = 0.0
    for i, (images, labels) in test_loader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size))
        counter += 1
        train_accuracy_sum = train_accuracy_sum + accuracy
    train_accuracy_ave = train_accuracy_sum / float(counter)

    # Test Loss
    counter = 0
    test_accuracy_sum = 0.0
    for i, (images, labels) in test_loader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum()) / float(batch_size))
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum / float(counter)

    curr_time = time.perf_counter()
    print("Epoch: {}/{}, Loss:{:.4f}, train accuracy:{:.4f}, test accuracy:{:4f}, time used:{:.4f}".
          format(epoch + 1, Num_Epochs, train_accuracy_ave, test_accuracy_ave,
                 curr_time - start_time))

    # save model
    torch.save(model.state_dict(), path_save)
