from pyp2p.net import *
import time

desktop = Net(passive_bind="192.168.0.45", passive_port=44444, interface="eth0:2", node_type="passive", debug=1)
desktop.start()
desktop.bootstrap()
desktop.advertise()

#Event loop.
while 1:
    for con in desktop:
        for reply in con:
            print(reply)

    time.sleep(1)
