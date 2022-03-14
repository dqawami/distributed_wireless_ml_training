import argparse
import socket

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for ML training')

    parser.add_argument('inet_type', 
    help='Either client or server', 
    type=str)

    parser.add_argument('ip_address', 
    help='IP address of server', 
    type=str)

    parser.add_argument('socket', 
    help='Socket number', 
    type=int)

    parser.add_argument('batch_size', 
    help='Batch size for this node', 
    type=int)

    parser.add_argument('--data_dir', 
    help='Data directory', 
    type=str, 
    default='./data')

    parser.add_argument('--save_data', 
    help='Whether to save data', 
    action='store_true')

    parser.add_argument('--buffer', 
    help='Size of buffer', 
    type=int, 
    default=4096)

    parser.add_argument('--train_workers', 
    help='Number of workers for training loader', 
    type=int, 
    default=2)

    parser.add_argument('--test_workers', 
    help='Number of workers for testing loader', 
    type=int, 
    default=2)

    parser.add_argument('--lr', 
    help='Model learning rate', 
    type=float, 
    default=0.001)

    parser.add_argument('--momentum', 
    help='Model momentum', 
    type=float, 
    default=0.9)

    parser.add_argument('--epochs', 
    help='Number of epochs for the model', 
    type=int, 
    default=16)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    assert args.inet_type == 'client' or args.inet_type == 'server'

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, 
                                            download=args.save_data, 
                                            transform=transform)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, 
                                           download=args.save_data, 
                                           transform=transform)

    net_node = socket.socket()
    net_details = (args.ip_address, args.socket)

    if args.inet_type == 'server':
        net_node.bind(net_details)
        net_node.listen(5)
        temp, _ = net_node.accept()
        net_node.close()
        net_node = temp
    else:
        net_node.connect(net_details)

    net_node.send(str(args.epochs).encode())
    other_epochs = int(net_node.recv(args.buffer).decode())
    assert args.epochs == other_epochs

    net_node.send(str(args.batch_size).encode())
    other_batch_size = int(net_node.recv(args.buffer).decode())

    total_batch = args.batch_size + other_batch_size

    print("Total batch", total_batch) 

    subset = (len(trainset) * args.batch_size) // total_batch
    trainset, _ = torch.utils.data.random_split(trainset, 
                                                [subset, 
                                                 len(trainset) - subset])

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=args.train_workers)
    
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.test_workers)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            net_node.send(str(loss.item()).encode())
            loss += float(net_node.recv(args.buffer).decode())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
   

if __name__ == '__main__':
    main()
