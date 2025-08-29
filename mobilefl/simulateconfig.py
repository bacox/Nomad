from typing import List
from mobilefl.server import Server
class SimulateConfig:
    def __init__(
        self,
        servers: List[Server],
        target_server: Server,
        move_from_server_id: int,
        move_client_id: int,
    ):
        self.servers = servers
        self.target_server = target_server
        self.move_from_server_id = move_from_server_id
        self.move_from_server = servers[move_from_server_id]
        self.move_client_id = move_client_id
        self.move_client = self.move_from_server.clients[move_client_id]
        self.target_client_dataset = self.move_from_server.client_train_datasets[move_client_id]
        self.move_from_server_dataset = self.move_from_server.client_train_datasets
        self.server_dataset = target_server.client_train_datasets
    def __str__(self) -> str:
        return (
            f"  - Move from server with ID: {self.move_from_server_id}\n"
            f"  - The ID of moving client of Server {self.move_from_server_id}: {self.move_client_id}\n"
            f"  - Target server ID: {self.target_server.server_id}\n"
        )
