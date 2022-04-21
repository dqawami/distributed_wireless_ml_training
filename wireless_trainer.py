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
        '''sendable_data = str(data)

        num_packets = int(math.ceil(len(sendable_data) / self.buffer))
        print(str(num_packets).encode())
        print(str(num_packets).encode().decode())

        self.net_node.send((str(num_packets) + ' ').encode())

        for i in range(num_packets):
            end = min(self.buffer * (i + 1), len(sendable_data))
            temp_data = sendable_data[(self.buffer * i):end]
            self.net_node.send(temp_data.encode())'''
        self.net_node.send(str(data).encode())

    def recv(self, cast_type):
        '''received_data = self.net_node.recv(self.buffer).decode()
        num_packets = received_data.split()[0]
        received_data = received_data.split(' ', 1)[1]
        print(num_packets)
        num_packets = int(num_packets)
        print(num_packets)

        if received_data != '':
            num_packets -= 1

        for _ in range(num_packets):
            received_data += self.net_node.recv(self.buffer).decode()

        return cast_type(received_data)'''
        return cast_type(self.net_node.recv(self.buffer).decode())

    def send_and_recv(self, data):
        self.send(data)
        return self.recv(type(data))

    def get_data_subset(self, trainset, splitter):
        subset = (len(trainset) * self.batch_size) // self.total_batch
        trainset, _ = splitter(trainset, [subset, len(trainset) - subset])
        return trainset

    def recursive_grad_averaging(self, layer, buffer=[]):
        out = []
        for l in layer:
            if type(l) == list:
                temp, buffer = self.recursive_grad_averaging(l, buffer)
                out.append(temp)
            else:
                temp = str(l) + ' '
                if not buffer:
                    other = self.send_and_recv(temp)
                    buffer.extend(other.split(' ')[:-1])
                out.append((l * self.batch_size + float(buffer.pop(0)) * self.other_batch)
                            / self.total_batch)
        return out, buffer

    def grad_averging(self, model):
        buffer = []
        for p in model.parameters():
            print("Before load")
            weights, buffer = self.recursive_grad_averaging(p.grad.tolist(), buffer)
            p.grad = torch.tensor(weights)
            print("Loaded parameter")
