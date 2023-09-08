import flwr as fl
from flwr.server.strategy import FedAvg
from client_manager import FedCSClientManager
from client import FlowerClient

def client_fn(cid: str):
    # Return a standard Flower client
    return FlowerClient(theta_mean=5, computation_mean=10, r=0.1)

C = 0.1
num_clients=10

T_round = 600
num_rounds = int(400/(T_round/60))

fl.common.logger.configure(identifier="FedCS_log", filename="log.txt")

# Launch the simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=num_clients, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=num_rounds), # Specify number of FL rounds
    strategy=FedAvg(fraction_fit=C, min_fit_clients=int(C*num_clients), min_available_clients=num_clients), # A Flower strategy
    client_manager=FedCSClientManager()
)