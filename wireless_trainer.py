import socket

class WirelessTrainer:
    def __init__(self, inet_type, ip_address: str, sock: int, buffer: int, 
                 epochs: int, batch_size: int, criterion, wait=5):
        assert inet_type == 'client' or inet_type == 'server'

        self.buffer = buffer
        self.batch_size = batch_size
        self._criterion = criterion

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
        self.net_node.send(str(data).encode())

    def recv(self, cast_type):
        return cast_type(self.net_node.recv(self.buffer).decode())

    def send_and_recv(self, data):
        self.send(data)
        return self.recv(type(data))

    def get_data_subset(self, trainset, splitter):
        subset = (len(trainset) * self.batch_size) // self.total_batch
        trainset, _ = splitter(trainset, [subset, len(trainset) - subset])
        return trainset

    def criterion(self, outputs, labels):
        loss = self._criterion(outputs, labels)
        other_loss = self.send_and_recv(loss.item())

        total_loss = (loss * self.batch_size + 
                      other_loss * self.other_batch) / self.total_batch

        return total_loss