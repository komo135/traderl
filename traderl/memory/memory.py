import torch
import numpy as np


class Memory:
    r"""
    A class to store the transitions that the agent observes, allowing us to reuse this data later.
    This class behaves like a cyclic buffer of bounded size that holds the transitions observed by the agent.
    It also implements a .sample() method for selecting a random batch of transitions for training.

    ## Example
    ```python
    import torch
    from traderl.memory import Memory

    batch_size = 64
    capacity = 10000
    state_shape = (30, 5)
    trading_state_shape = (30, 5)
    action_shape = 1
    action_type = torch.int32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    memory = Memory(capacity, state_shape, trading_state_shape, action_shape, action_type, device)
    ```
    """

    def __init__(self, capacity: int, state_shape: tuple, trading_state_shape: tuple,
                 action_shape: int, action_type: torch.dtype, device: torch.device):
        r"""
        Initialize a new instance of the Memory class.

        ## Args
        - capacity: The maximum number of transitions that the memory can store.
        - state_shape: The shape of the state. example: (30, 5)
        - trading_state_shape: The shape of the trading state. example: (30, 5)
        - action_shape: The shape of the action. example: 1
        - action_type: The type of the action. example: torch.int32 or torch.float32
        - device: The device to send tensors to.
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.trading_state_shape = trading_state_shape
        self.action_shape = action_shape
        self.action_type = action_type
        self.device = device

        # Initialize memory arrays
        self.states = torch.empty((self.capacity, *self.state_shape), dtype=torch.float32, device=self.device)
        self.trading_states = torch.empty((self.capacity, *self.trading_state_shape),
                                          dtype=torch.float32, device=self.device)
        self.new_states = torch.empty((self.capacity, *self.state_shape), dtype=torch.float32, device=self.device)
        self.new_trading_states = torch.empty((self.capacity, *self.trading_state_shape),
                                              dtype=torch.float32, device=self.device)
        self.actions = torch.empty((self.capacity, self.action_shape), dtype=self.action_type, device=self.device)
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.device)

        self.index = 0
        self.full = False

    def __len__(self):
        """Return the current size of internal memory."""
        return self.capacity if self.full else self.index

    def append(self, state, trading_state, action, reward, new_state, new_trading_state, done):
        r"""
        Add a new transition to memory.

        ## Args
        - state: The state of the environment.
        - trading_state: The state of the trading environment.
        - action: The action taken by the agent.
        - reward: The reward received by the agent.
        - new_state: The new state of the environment.
        - new_trading_state: The new state of the trading environment.
        - done: Whether the episode is finished or not.

        ## Example
        ```python=
        memory.append(state, trading_state, action, reward, new_state, new_trading_state, done)
        ```
        """

        self.trading_states[self.index] = trading_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index] = new_state
        self.new_trading_states[self.index] = new_trading_state
        self.dones[self.index] = done

        self.index = self.index + 1 if self.index + 1 < self.capacity else 0
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Randomly sample a batch of transitions from memory.

        ## Args
        - batch_size: The number of transitions to sample.

        ## Returns
        A tuple of (states, trading_states), actions, rewards, (new_states, new_trading_states), dones.

        ## Example
        ```python=
        states, actions, rewards, new_states, dones = memory.sample(batch_size)
        ```
        """
        indices = np.unique(np.random.randint(0, len(self), batch_size*2))[:batch_size]

        states = self.states[indices]
        trading_states = self.trading_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        new_states = self.new_states[indices]
        new_trading_states = self.new_trading_states[indices]
        dones = self.dones[indices]

        return (states, trading_states), actions, rewards, (new_states, new_trading_states), dones
