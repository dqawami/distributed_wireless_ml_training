import argparse
import socket

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for ML training')

    parser.add_argument('ip_address', 
    help='IP address of server', 
    type=str)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    client = socket.socket()
    client.connect((args.ip_address, 8080))

    # client.send("I am CLIENT\n".encode())
    client.send(str(3.444).encode())

    from_server = client.recv(4096)

    client.close()

    print(float(from_server.decode()))

if __name__ == '__main__':
    main()
