import math
import socket
import torch

class WirelessTrainer:
    def __init__(self, inet_type, ip_address: str, sock: int, buffer: int, 
                 epochs: int, batch_size: int, wait=5):
        assert inet_type == 'client' or inet_type == 'server'

        self.buffer = buffer
        self.batch_size = batch_size

        self.net_node = socket.socket()
        net_details = (ip_address, sock)

        self.receive_out = []

        if inet_type == 'server':
            self.net_node.bind(net_details)
            self.net_node.listen(wait)
            temp, _ = self.net_node.accept()
            self.net_node.close()
            self.net_node = temp
        else:
            self.net_node.connect(net_details)

        assert epochs == self.send_and_recv(epochs)

        self.other_batch = self.send_and_recv(batch_size)

        self.total_batch = batch_size + self.other_batch

    def send(self, data):
        send_data = str(data) + ' '
        self.net_node.send(send_data.encode())

    def recv(self, cast_type):
        if not self.receive_out:

            self.receive_out.extend(self.net_node.recv(self.buffer).decode().split(' ')[:-1])
        return cast_type(self.receive_out.pop(0))

    def send_and_recv(self, data):
        self.send(data)
        return self.recv(type(data))

    def get_data_subset(self, trainset, splitter):
        subset = (len(trainset) * self.batch_size) // self.total_batch
        trainset, _ = splitter(trainset, [subset, len(trainset) - subset])
        return trainset

    def recursive_grad_averaging(self, grad):
        out = []
        for g in grad:
            if type(g) != list:
                other = self.send_and_recv(g)
                out.append(type(g)((g * self.batch_size + other * self.other_batch) / self.total_batch))
            else:
                out.append(self.recursive_grad_averaging(g))

        return out

    def grad_averaging(self, model):
        for param in model.parameters():
            param.grad.data = torch.tensor(self.recursive_grad_averaging(
                param.grad.data.tolist()))
