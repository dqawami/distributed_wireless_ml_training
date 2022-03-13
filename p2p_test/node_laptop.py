from pyp2p.net import *

laptop = Net(passive_bind="192.168.0.44", passive_port=44445, interface="eth0:1", node_type="passive", debug=1)
laptop.start()
laptop.bootstrap()
laptop.advertise()

#Event loop.
while 1:
    for con in laptop:
        con.send_line("test")

    time.sleep(1)
