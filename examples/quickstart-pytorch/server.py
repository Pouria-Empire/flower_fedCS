from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from client_manager import FedCSClientManager


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


C = 0.1

# Define strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=C, min_fit_clients=5, min_available_clients=50, evaluate_metrics_aggregation_fn=weighted_average)

#
client_manager = FedCSClientManager()

# Start Flower server
fl.server.start_server(
    server_address="192.168.0.1:8084",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    client_manager=client_manager
)
