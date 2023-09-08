import random
import threading
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
    ReconnectIns,
)

from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from flwr.server.client_manager import SimpleClientManager

# from flower.src.py.flwr.server.client_proxy import ClientProxy
# from flower.src.py.flwr.server.client_manager import SimpleClientManager

class FedCSClientManager(SimpleClientManager):
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        log(
                INFO,
                "num_clients = (%s), min_num_clients = (%s).",
                num_clients,
                min_num_clients
            )
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        available_clients = [self.clients[cid] for cid in sampled_cids]

        theta = {}
        t_ud = {}
        for client in available_clients:
            getParametersRes = client.get_properties(GetPropertiesIns(config={'gpu':0}), 1000)
            resources = getParametersRes.properties
            theta[client.cid] = resources['theta']
            t_ud[client.cid] = resources['t_ud']

        D_m = 115.2 # model size 115.2 or 146.4
        T_round = 600 # round deadline

        T_d = 0
        T_d_new = 0
        Theta = 0
        Theta_new = 0
        K = sampled_cids
        S = []
        while K:
            max_val = -1
            argmax = -1
            for cid in K:
                T_d_new = max(T_d, D_m/theta[cid])
                t_ul = D_m/theta[cid]
                val = T_d_new - T_d + t_ul + max(0, t_ud[cid] - Theta)
                if val > max_val:
                    max_val = val
                    argmax = cid
            K.remove(argmax)
            t_ul = D_m/theta[argmax]
            Theta_new  = Theta + t_ul + max(0, t_ud[argmax] - Theta)
            T_d_new = max(T_d, D_m/theta[argmax])
            t = T_d_new + Theta_new
            if t < T_round:
                Theta = Theta_new
                T_d = T_d_new
                S.append(argmax)

        log(
                INFO,
                'Theta = (%s), T_d = (%s) and The following clients were selected for this round: (%s)',
                Theta,
                T_d,
                S
            )

        return [self.clients[cid] for cid in S]