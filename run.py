from model.dataloader import train_loader, test_loader

from model.net import CnnNet_small
import torch

from spikingjelly.activation_based import encoding
from model.train_function import Train_func1

# 实例化网络

net = CnnNet_small()
net.initialize()

lr = 1e-4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = 'cpu'
net.to(device)

# 使用Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 使用泊松编码器
encoder = encoding.PoissonEncoder()

Epoch = 100

T = 50

test = True

for epoch in range(Epoch):

    print('training and testing on :', device)

    if test == True:

        train_loss, train_acc, train_time, test_acc = Train_func1(net, train_loader, test_loader, optimizer, device, encoder, test)

        print('train_loss', train_loss, 'epoch', epoch, 'train_acc', train_acc, 'train_time', train_time)

        print('test_acc', test_acc)

    else:

        train_loss, train_acc, train_time = Train_func1(net, train_loader, test_loader, optimizer, device,
                                                                  encoder, test)

        print('train_loss', train_loss, 'epoch', epoch, 'train_acc', train_acc, 'train_time', train_time)