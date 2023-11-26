import numpy as np
from collections import deque
import torch

from traderl.environment import Env
from traderl.memory import Memory
from traderl.model import build_model
from common import Evolution


class DQN:
    """
    ## Deep Q-Network (DQN) Agent

    This class implements a standard DQN agent which is suitable for discrete action spaces. 
    It utilizes a replay buffer for experience replay and a target network to stabilize training.

    ### Attributes:
    - `name` (str): Name of the agent.
    - `actor_critic` (bool): Flag to indicate if the model is actor-critic (False for standard DQN).
    - `model` (torch.nn.Module): The Q-network model.
    - `target_model` (torch.nn.Module): The target Q-network for stable Q targets.
    - `i` (int): Internal counter to track update steps.

    ### Example:
    ```python
    dqn_agent = DQN(
        network_name='convnet',
        action_space=3,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        replay_buffer_size=1e6,
        replay_ratio=4,
        target_update=100,
        tau=0.01,
        batch_size=64,
        n_step=4
    )
    ```
    """

    name = "dqn"
    actor_critic = False

    # Models will be initialized in the build_model method
    model: torch.nn.Module
    target_model: torch.nn.Module

    i = 0  # Initialize update counter

    def __init__(self,
                 network_name: str,
                 action_space: int = 3,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 replay_buffer_size: int = 1e6,
                 replay_ratio: int = 4,
                 target_update: int = 100,
                 tau: float = 0.01,
                 batch_size: int = 64,
                 n_step: int = 4,
                 load: bool = False):
        """
        ### Initializes the DQN agent.

        #### Parameters:
        - `network_name` (str): The name of the neural network architecture to use.
        - `action_space` (int): The number of actions available in the action space.
        - `learning_rate` (float): The learning rate for training the Q-network.
        - `gamma` (float): The discount factor for future rewards.
        - `epsilon` (float): The exploration rate for epsilon-greedy action selection.
        - `replay_buffer_size` (int): The size of the replay buffer.
        - `replay_ratio` (int): The ratio of learning updates to environment steps.
        - `target_update` (int): The frequency (in steps) at which the target network is updated.
        - `tau` (float): The soft update parameter for blending the target network parameters.
        - `batch_size` (int): The batch size for sampling from memory.
        - `n_step` (int): The number of steps for n-step Q learning.
        - `load` (bool): Whether to load an existing model or not.

        #### Outputs:
        None. Initializes internal variables and neural network models.
        """

        self.network_name = network_name
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.target_update = target_update
        self.tau = tau
        self.batch_size = batch_size
        self.n_step = n_step
        self.load = load

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.symbols, self.data = self.get_data()
        self.train_step, self.test_step = self._split_data()

        self.env = Env("discrete", self.data, self.device)
        self.test_env = Env("discrete", self.data, self.device)

        self.evolution = Evolution(self.test_env)

        self.save_path = f"traderl/agent/{self.name}.pt"
        if self.load:
            self.load_agent()
        else:
            self.memory = Memory(
                self.replay_buffer_size,
                self.data["state"].shape[-2:],
                self.env.trade_state.shape[-2:],
                1, torch.int32, self.device
            )
            self.build_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _split_data(self):
        """
        ### Splits the data into training and testing sets.

        #### Outputs:

        """
        length = len(self.data["state"])
        total_steps = range(length)
        test_step = total_steps[int(length * 0.9):]
        train_step = total_steps[:test_step[0]]

        return train_step, test_step

    def _initialize_deques(self):
        """
        ### Initializes and returns a list of deques for storing states, trading states, actions, rewards, and dones.

        #### Outputs:
        - `deques` (list[deque]): A list of deques, each with a maximum length of `self.n_step + 1`.

        #### Example:
        ```python
        states, trading_states, actions, rewards, dons = dqn_agent._initialize_deques()
        ```
        """
        return [deque(maxlen=self.n_step + 1) for _ in range(5)]

    def _get_start_end(self):
        start_or_end = np.random.randint(0, 2)
        period = 50000
        if start_or_end == 0:
            start = np.random.randint(0, self.train_step[-1] - period)
            end = start + period
        else:
            end = np.random.randint(period, self.train_step[-1])
            start = end - period

        return start, end

    def build_model(self):
        r"""
        ### Builds the Q-network and target Q-network models.

        #### Outputs:
        None. Initializes `self.model` and `self.target_model` as instances of the neural network.
        """
        self.model = build_model(
            self.network_name,
            self.action_space,
            self.data["state"].shape[-2:],
            self.actor_critic
        )
        self.target_model = build_model(
            self.network_name,
            self.action_space,
            self.data["state"].shape[-2:],
            self.actor_critic
        )
        self.target_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.target_model.to(self.device)

    def get_action(self, state: tuple[torch.tensor, torch.tensor]) -> int:
        """
        ### Returns an action based on the current state.

        #### Parameters:
        - `state` (tuple[torch.tensor, torch.tensor]): The current state of the environment.

        #### Outputs:
        - `action` (int): The action to take.

        #### Example:
        ```python
        action = dqn_agent.get_action(state)
        ```
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            with torch.no_grad():
                action = self.model(*state).max(1)[1].item()
        return action

    @staticmethod
    def get_data() -> tuple[int, dict]:
        """
        ### Loads and returns trading data used by the environment.

        #### Outputs:
        - `symbols` (int): The number of trading symbols.
        - `data` (dict): A dictionary containing different price information arrays.
          - `state`: The state information array.
          - `open`: The opening price array.
          - `high`: The high price array.
          - `low`: The low price array.
          - `atr`: The average true range array.

        #### Example:
        ```python
        symbols, data = DQN.get_data()
        ```
        """
        data = np.load('traderl/data/data.npz')
        data_dict = {
            'state': data['state'],
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'atr': data['atr']
        }

        symbols = data_dict['state'].shape[0]

        return symbols, data_dict

    def load_agent(self):
        """
        ### Loads the entire agent from the specified path.

        #### Outputs:
        None. Loads the agent from disk.
        """
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['memory']
        self.epsilon = checkpoint['epsilon']
        self.i = checkpoint['i']

    def save_agent(self):
        """
        ### Saves the entire agent to the specified path.

        #### Outputs:
        None. Saves the agent to disk.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory,
            'epsilon': self.epsilon,
            'i': self.i
        }, self.save_path)

    def train(self, num_iterations=1000000):
        states, trading_states, actions, rewards, dons = self._initialize_deques()
        start, end = self._get_start_end()
        step = self.env.step(self.get_action, start, end, True)

        done = 1

        for i in range(num_iterations):
            returns = next(step, None)
            if returns is None or done == 0:
                print(f"symbol: {self.env.symbol}, start: {start}, end: {end}")
                print(f"total pip: {self.env.total_pip}, asset: {self.env.asset}")
                states, trading_states, actions, rewards, dons = self._initialize_deques()

                if (i + 1) % 15 == 0:
                    start, end = self._get_start_end()
                else:
                    self.env.symbol -= 1
                step = self.env.step(self.get_action, start, end, True)
            else:
                state, trading_state, action, reward, done = returns

                states.append(state)
                trading_states.append(trading_state)
                actions.append(action)
                rewards.append(reward)
                dons.append(done)

                if len(states) == self.n_step + 1:
                    # Calculate n_reward in multiple steps for clarity
                    n_reward = 0
                    for i in range(self.n_step - 1):
                        n_reward += self.gamma ** i * rewards[i]

                    n_state = states[-1]
                    n_trading_state = trading_states[-1]
                    n_done = dons[-1]

                    # Store the first state, trading state, action, and done flag in temporary variables
                    first_state = states[0]
                    first_trading_state = trading_states[0]
                    first_action = actions[0]

                    self.memory.append(first_state, first_trading_state, first_action,
                                       n_reward, n_state, n_trading_state, n_done)

                    # Add a comment to explain the condition for calling self.update()
                    # Update the network parameters if the current index of the memory is divisible by the replay ratio
                    # and the length of the memory is greater than the batch size
                    if self.memory.index % self.replay_ratio == 0 and len(self.memory) > self.batch_size * 10:
                        self.update()

                    if self.i % 1000 == 0:
                        self.evolution.evolute(self.get_action, self.test_step[0], self.test_step[-1])

            if (i + 1) % 100000 == 0:
                self.save_agent()

    def update(self):
        """
        ### Performs a single step of optimization using a batch from the replay buffer.

        #### Outputs:
        None. Updates the `self.model` parameters based on the computed loss.
        """
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        q_values = self.model(*state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.model(*next_state).max(1)[1]
            next_q_values = self.target_model(*next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)

        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = (q_values - expected_q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.i += 1

        self.update_target_model()

    def update_target_model(self):
        r"""
        ### Softly updates the target network parameters based on `tau`.

        #### Outputs:
        None. Modifies `self.target_model` parameters in place.
        """
        if self.i % self.target_update == 0:
            with torch.no_grad():
                updated_state_dict = {
                    name: self.tau * self.model.state_dict()[name] +
                    (1 - self.tau) * self.target_model.state_dict()[name]
                    for name in self.model.state_dict()
                }
                self.target_model.load_state_dict(updated_state_dict)
