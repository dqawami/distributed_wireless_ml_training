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

    serv = socket.socket()

    serv.bind((args.ip_address, 8080))
    serv.listen(5)

    print("Client searched")
    conn, addr = serv.accept()
    print("Client accepted!")
    from_client = ''

    while True:
        data = conn.recv(4096)
        if not data: break
        from_client += data.decode()
        print(from_client)
        conn.send("I am SERVER\n".encode())
        
    conn.close()
    print('client disconnected')

if __name__ == '__main__':
    main()
