import torch
from traderl.model import all_models

__all__ = ["build_model"]


class NetworkKeyNotFoundError(Exception):
    """Raised when the network key is not found in the all_models dictionary"""
    pass


class QModel(torch.nn.Module):
    def __init__(self, network_name, action_space, in_channels):
        super().__init__()
        self.network_name = network_name
        self.action_space = action_space
        self.in_channels = in_channels

        self.network1 = all_models[self.network_name](self.in_channels)
        self.network2 = all_models[self.network_name](self.in_channels)

        self.fc = self.create_fc_layers()

    def create_fc_layers(self):
        fc = torch.nn.Sequential(
            torch.nn.Linear(self.network1.out_channels * 2, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, self.action_space),
        )
        return fc

    def forward(self, x1, x2):
        x1 = self.network1(x1)
        x2 = self.network2(x2)
        x = torch.cat((x1, x2), dim=-1)
        return self.fc(x)


class ActorModel(torch.nn.Module):
    def __init__(self, network_name, action_space, in_channels):
        super().__init__()
        self.network_name = network_name
        self.action_space = action_space
        self.in_channels = in_channels

        self.network = all_models[self.network_name](self.in_channels)

        self.fc = self.create_fc_layers()

    def create_fc_layers(self):
        fc = torch.nn.Sequential(
            torch.nn.Linear(self.network.out_channels, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, self.action_space),
            torch.nn.Tanh(),
        )
        return fc

    def forward(self, x):
        x = self.network(x)
        return self.fc(x)


def build_model(network_name, action_space, in_channels, actor_critic=False) -> torch.nn.Module | tuple[torch.nn.Module, torch.nn.Module]:
    """
    Build a model.

    Args:
        network_name (str): Name of the network.
        action_space (int): Number of actions.
        in_channels (int): Number of input channels.
        actor_critic (bool): Whether to build an actor-critic model or not.

    Returns:
        torch.nn.Module | tuple[torch.nn.Module, torch.nn.Module]: The model or a tuple of actor and critic models.
    """
    try:
        if actor_critic:
            actor = ActorModel(network_name, action_space, in_channels)
            critic = QModel(network_name, 1, in_channels)
            return actor, critic
        else:
            model = QModel(network_name, action_space, in_channels)
            return model
    except KeyError:
        raise NetworkKeyNotFoundError(f"Network {network_name} not found in the all_models dictionary")
