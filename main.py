import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.optim as optim

from wireless_trainer import WirelessTrainer

from model_wrapper import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for ML training')

    parser.add_argument('model', 
    help='Model type', 
    type=str)

    parser.add_argument('--data_dir', 
    help='Data directory', 
    type=str, 
    default='./data')

    parser.add_argument('--save_data', 
    help='Whether to save data', 
    action='store_true')

    parser.add_argument('--cuda', 
    help='Enables CUDA (if available',
    action='store_true')

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

    parser.add_argument('--batch_size', 
    help='Batch size for this node', 
    type=int, 
    default=10)

    parser.add_argument('--mini_batch', 
    help='Mini-batch to report loss', 
    type=int, 
    default=2000)

    parser.add_argument('--epochs', 
    help='Number of epochs for the model', 
    type=int, 
    default=200)

    parser.add_argument('--wireless', 
    help='Whether or not training is wireless', 
    action='store_true')

    parser.add_argument('--inet_type', 
    help='Either client or server', 
    type=str, 
    default=None)

    parser.add_argument('--ip_address', 
    help='IP address of server', 
    type=str, 
    default=None)

    parser.add_argument('--socket', 
    help='Socket number', 
    type=int, 
    default=-1)

    parser.add_argument('--buffer', 
    help='Size of buffer', 
    type=int, 
    default=4096)

    parser.add_argument('--checkpoint_dir',
    help='Directory to save model checkpoints',
    type=str,
    default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

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

    criterion = nn.CrossEntropyLoss()

    if args.wireless:
        assert (args.inet_type is not None and args.ip_address is not None
                and args.socket > 0 and args.buffer > 0)
        wireless_trainer = WirelessTrainer(args.inet_type, args.ip_address, 
                                        args.socket, args.buffer, args.epochs, 
                                        args.batch_size, criterion)

        print("Total batch", wireless_trainer.total_batch) 

        trainset = wireless_trainer.get_data_subset(trainset, 
                                                    torch.utils.data.random_split)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=args.train_workers)
    
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.test_workers)

    net = get_model(args.model)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    if args.checkpoint_dir is None:
        checkpoint_dir = './checkpoints/' + args.model
    else:
        checkpoint_dir = args.checkpoint_dir

    folders = checkpoint_dir.split('/')

    temp = folders.pop(0) + '/'
    for f in folders:
        if not os.path.isdir(temp):
            os.mkdir(temp)
        temp += f + '/'

    if args.cuda and torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if args.wireless:
                loss = wireless_trainer.criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.mini_batch == (args.mini_batch - 1):
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / args.mini_batch:.3f}')
                running_loss = 0.0

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

        acc = 100 * correct // total
        print(f'Accuracy of the network at Epoch {epoch + 1}: {acc} %')

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
        }
        torch.save(state, checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth')

    print('Finished Training')
   

if __name__ == '__main__':
    main()
