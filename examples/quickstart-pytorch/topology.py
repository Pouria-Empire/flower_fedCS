from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI

class CustomTopology(Topo):
    def build(self, n=2):
        # Create the server
        server = self.addHost('server', ip='192.168.0.1')

        # Create the switch
        switch = self.addSwitch('s1')

        # Create n clients using a loop
        clients = []
        for i in range(2, n + 2):
            client = self.addHost(f'client{i}', ip=f'192.168.0.{i}')
            clients.append(client)
            self.addLink(client, switch)

        # Link the server to the switch
        self.addLink(server, switch)

def main():
    num_clients = 50
    topo = CustomTopology(n=num_clients)
    net = Mininet(topo=topo)
    net.start()

    # Start server.py script on the server host
    server = net.get('server')
    server.cmd('/home/user01/test/flower_start/bin/python3 server.py > ./server_log/server_log.txt 2>&1 &')

    # Start client.py scripts on the client hosts using a loop
    for i in range(2, num_clients+2):  # Replace 3 with the number of clients you want
        client = net.get(f'client{i}')
        client.cmd(f'/home/user01/test/flower_start/bin/python3 client_test.py > ./clients_log/client{i-1}_log.txt 2>&1 &')# > ./clients_log/client_log.txt 2>&1 &')


    net.monitor()

    # Open Mininet CLI for interaction
    CLI(net)

    # Clean up and stop the network
    net.stop()

if __name__ == '__main__':
    main()
