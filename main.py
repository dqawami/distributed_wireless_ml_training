import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt

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

    parser.add_argument('--resume', 
    help='Resumes from a checkpoint', 
    type=str, 
    default=None)

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
    default=0.01)

    parser.add_argument('--momentum', 
    help='Model momentum', 
    type=float, 
    default=0.9)

    parser.add_argument('--batch_size', 
    help='Batch size for this node', 
    type=int, 
    default=10)

    parser.add_argument('--super_batch', 
    help='Super batch for this node (bigger than the parser batch', 
    type=int, 
    default=1)

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

    parser.add_argument('--plot_name', 
    help='Name of the accuracy plot (saved in the checkpoint_dir)', 
    type=str, 
    default='plot.png')

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

    net = get_model(args.model)

    hook = None

    if args.wireless:
        assert (args.inet_type is not None and args.ip_address is not None
                and args.socket > 0 and args.buffer > 0)
        wireless_trainer = WirelessTrainer(args.inet_type, args.ip_address, 
                                        args.socket, args.buffer, args.epochs, 
                                        args.batch_size)

        print("Total batch", wireless_trainer.total_batch) 

        trainset = wireless_trainer.get_data_subset(trainset, 
                                                    torch.utils.data.random_split)
        hook = wireless_trainer.grad_averaging

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=args.train_workers)
    
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             num_workers=args.test_workers)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    if args.checkpoint_dir is None:
        checkpoint_dir = './checkpoints/' + args.model
    else:
        checkpoint_dir = args.checkpoint_dir

    folders = checkpoint_dir.split('/')

    temp = folders.pop(0) + '/'
    folders.append('')
    for f in folders:
        if not os.path.isdir(temp):
            os.mkdir(temp)
        temp += f + '/'

    plot_name = checkpoint_dir + '/' + args.plot_name

    start_epoch = 0
    epochs = []
    accs = []

    if args.resume is not None:
        assert os.path.exists(args.resume), 'Error: checkpoint file not found'
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        epochs.append(checkpoint['epoch'])
        accs.append(checkpoint['acc'])

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')

    super_batch = max(1, args.super_batch)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - start_epoch + super_batch - 1))

    net.to(device)

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        super_count = 1
        loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = net(inputs.to(device), hook)
            loss += criterion(outputs, labels)

            # print statistics
            running_loss += loss.item() / super_count

            if super_count == super_batch:
                optimizer.zero_grad()
                loss.backward()
                '''if args.wireless:
                    wireless_trainer.grad_averging(net)'''
                optimizer.step()
                super_count = 0

                loss = 0.0

            if i % args.mini_batch == (args.mini_batch - 1):
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / args.mini_batch :.3f}')
                running_loss = 0.0

            super_count += 1

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images.to(device))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct // total
        print(f'Accuracy of the network at Epoch {epoch + 1}: {acc} %')

        epochs.append(epoch + 1)
        accs.append(acc)

        plt.clf()
        plt.plot(epochs, accs)
        plt.savefig(plot_name)

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
        }
        torch.save(state, checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth')

        scheduler.step()

    print('Finished Training')
   

if __name__ == '__main__':
    main()
