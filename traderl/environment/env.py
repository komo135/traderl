import numpy as np
import torch


class Env:
    """
    This class represents the trading environment.
    """
    initial_asset = 100000  # Initial asset in Japanese yen
    risk = 0.05  # Risk per trade
    spread = 10  # Spread

    sim_limit = 10000  # Simulation limit
    sim_stop_cond = 0.1  # Simulation stop condition (0.75 means 75% of max asset)

    def __init__(self, action_type: str, data: dict, device: torch.device):
        """
        Initialize the trading environment.

        Parameters
        ----------
        action_type : str
            The type of action, either 'discrete' or 'continuous'.
        data : dict
            The data of stock/forex price. It should contain 'state', 'open', 'high', 'low', 'atr'.
            The shapes should be (symbols, length).
        device : torch.device
            The device to use for computations.
        """
        self.action_type = action_type
        self.data = data
        self.device = device

        self.state = data['state']
        self.open = data['open']
        self.high = data['high']
        self.low = data['low']
        self.atr = data['atr']

        self.min_stop_losses = np.array([np.median(atr) for atr in self.atr])
        self.max_stop_losses = np.array([np.quantile(atr, 0.99) for atr in self.atr])

        self.symbols = self.open.shape[0]
        self.symbol = -1

        self.asset = self.initial_asset
        self.total_pip = 0
        self.pips, self.win_pips, self.lose_pips = [[] for _ in range(3)]
        self.profits = []
        self.actions = []
        self.max_asset, self.asset_drawdown = self.asset, 1.0
        self.trade_event = {"long": [], "short": []}

        self.trade_state = torch.empty((1, 4, 30), dtype=torch.float32, device=self.device)
        self.state = torch.tensor(self.state, dtype=torch.float32, device=self.device)

    def reset_env(self):
        """
        Reset the trading environment to its initial state.
        """
        self.asset = self.initial_asset
        self.total_pip = 0
        self.pips, self.win_pips, self.lose_pips = [[] for _ in range(3)]
        self.profits = []
        self.actions = []

        self.trade_state *= 0

        self.symbol += 1
        if self.symbol >= self.symbols:
            self.symbol = 0

    def reset_trade(self) -> tuple[float, float, int]:
        """
        Reset the trading environment to its initial state.

        Returns
        -------
        tuple
            A tuple containing the profit factor, expected ratio, and number of days.
        """
        self.asset = self.initial_asset
        self.pips, self.win_pips, self.lose_pips = [[] for _ in range(3)]
        self.profits = []
        self.max_asset, self.asset_drawdown = self.asset, 1.0

        self.trade_state *= 0

        profit_factor, expected_ratio = 0, 0
        days = 0

        return profit_factor, expected_ratio, days

    def start_trade(self, i, atr) -> tuple[float, int, int, float, int]:
        """
        Start a new trade.

        Parameters
        ----------
        i : int
            The index of the last step.
        atr : np.array
            The atr.

        Returns
        -------
        tuple
            A tuple containing the pip, old index, trade length, stop loss, and position size.
        """
        pip = 0
        old_i = i
        trade_length = 0
        stop_loss = np.clip(atr[i] * 1, self.min_stop_losses[self.symbol], self.max_stop_losses[self.symbol])
        position_size = int(self.asset * self.risk / stop_loss)
        position_size = np.minimum(np.maximum(position_size, 0), 10000000)

        return pip, old_i, trade_length, stop_loss, position_size

    def stop_trade(self, pip, action, position_size):
        """
        Stop the current trade.

        Parameters
        ----------
        pip : float
            The profit or loss.
        action : int
            The action of the last step.
        position_size : int
            The position size of the last step.
        """
        if action != 0:
            self.total_pip += pip
            self.asset += pip * position_size
            self.pips.append(pip)
            if pip > 0:
                self.win_pips.append(pip)
            else:
                self.lose_pips.append(pip)
            self.profits.append(self.asset)

            self.max_asset = np.maximum(self.max_asset, self.asset)
            self.asset_drawdown = self.asset / self.max_asset

    def get_metrics(self) -> tuple[float, float]:
        """
        Calculate the profit factor and expected ratio of the trading environment.

        Returns
        -------
        tuple
            A tuple containing the profit factor and expected ratio of the trading environment.
        """
        if len(self.pips) >= 20 and self.win_pips and self.lose_pips:
            profit_factor = np.sum(self.win_pips) / np.abs(np.sum(self.lose_pips))

            acc = np.mean(np.array(self.pips) > 0)
            win, lose = np.mean(self.win_pips) * acc, np.mean(self.lose_pips) * (1 - acc)
            expected_ratio = (win + lose) / np.abs(lose) + 1
        else:
            profit_factor = 0
            expected_ratio = 0

        return np.clip(profit_factor, 0, 2), np.clip(expected_ratio, 0, 2)

    def get_data(self, start_index: int, end_index: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the state, open, high, low, and atr arrays for the given start and end indices.

        Parameters
        ----------
        start_index : int
            The starting index of the data.
        end_index : int
            The ending index of the data.

        Returns
        -------
        tuple
            A tuple containing the state, open, high, low, and atr arrays.
        """
        state = self.state[self.symbol, start_index:end_index].clone()
        open = self.open[self.symbol, start_index:end_index]
        high = self.high[self.symbol, start_index:end_index]
        low = self.low[self.symbol, start_index:end_index]
        atr = self.atr[self.symbol, start_index:end_index]

        default_event = ["no event" for _ in range(len(state))]
        self.trade_event = {"open": default_event.copy(), "long": default_event.copy(), "short": default_event.copy()}

        return state, open, high, low, atr

    def update_trade_state(self, update_values: list, tentative_update=False):
        """
        Update the trade state.

        Parameters
        ----------
        update_values : list
            The values to update the trade state with.
        tentative_update : bool, optional
            Whether the update is tentative or not, by default False.
        """
        if not tentative_update:
            self.trade_state[0, :, :-1].copy_(self.trade_state[0, :, 1:])
            self.trade_state[0, :, -1].zero_()

        self.trade_state[0, :, -1].copy_(torch.tensor(np.round(update_values, 3)))

    def step(self, get_action, start_index: int, end_index: int, train=False):
        """
        Perform a step in the trading environment.

        Parameters
        ----------
        get_action : function
            The function to get the action.
        start_index : int
            The starting index of the data.
        end_index : int
            The ending index of the data.
        train : bool, optional
            Whether the step is for training or not, by default False.
        """
        self.reset_env()
        state, open, high, low, atr = self.get_data(start_index, end_index)
        profit_factor, expected_ratio, days = self.reset_trade()

        is_stop = True
        now_state = [0, 0]

        hit_point = 30
        zeros_days = 0

        for i in range(len(state) - 1):
            done, reward = 1, 0
            days += 1

            if is_stop:
                pip, old_i, trade_length, stop_loss, position_size = self.start_trade(i, atr)
                skip = 5

                # action: 1 -> buy(long position), -1 -> sell(short position), 0 -> hold(non position)
                now_state = [state[[i]], self.trade_state.clone()]
                if self.action_type == 'discrete':
                    policy = get_action(now_state, train=train)
                    action = 1 if policy == 0 else -1 if policy == 1 else 0
                    take_profit = stop_loss * 2
                else:
                    policy = get_action(now_state, train=train)
                    action = np.sign(policy)
                    take_profit = np.clip(stop_loss * np.abs(policy) * 2, 1, None)

                if action == 1:
                    self.trade_event["open"][i] = "long"
                elif action == -1:
                    self.trade_event["open"][i] = "short"

                add_hit_point = 0

            self.actions.append(action)
            add_hit_point -= 0.05

            if action == 0:
                skip -= 1
                is_stop = skip <= 0
            else:
                trade_length += 1
                pip = (open[i + 1] - open[old_i]) * action - self.spread

                higher_pip = (high[i] - open[old_i] if action == 1 else open[old_i] - low[i]) - self.spread
                lower_pip = (low[i] - open[old_i] if action == 1 else open[old_i] - high[i]) - self.spread

                now_higher_pip = (high[i] - open[i] if action == 1 else open[i] - low[i]) - self.spread
                now_lower_pip = (low[i] - open[i] if action == 1 else open[i] - high[i]) - self.spread

                event = None
                if lower_pip <= -stop_loss:
                    if ((open[i] - open[old_i]) * action - self.spread) >= 0:
                        print(f"pip: {pip}, lower_pip: {lower_pip} now_lower_pip:"
                              f"{now_lower_pip}, stop_loss: {stop_loss}, position_size: {position_size}, action: {action}"
                              f" symbol: {self.symbol}, old_i: {old_i}, i: {i}, start_index: {start_index}, end_index: {end_index}")

                    pip = -stop_loss
                    is_stop = True
                    event = "stop loss"
                elif higher_pip >= take_profit:
                    pip = take_profit
                    is_stop = True
                    event = "take profit"
                    add_hit_point = np.round(pip / stop_loss, 3)
                elif trade_length >= 50:
                    is_stop = True
                    event = "stop trade"
                else:
                    is_stop = False

                if event is not None:
                    if action == 1:
                        self.trade_event["long"][i] = event
                    elif action == -1:
                        self.trade_event["short"][i] = event

            if is_stop:
                hit_point = np.clip(np.round(hit_point + add_hit_point, 2), 0, 30)
                now_dyas = (days + 1) / self.sim_limit
                now_hp = hit_point / 30

                if self.trade_state[0, 2, -1] == 0 and action == 0:
                    add_hit_point += self.trade_state[0, -1, -1].item()
                    self.update_trade_state([now_dyas, now_hp, 0, add_hit_point], tentative_update=True)
                else:
                    self.update_trade_state([now_dyas, now_hp, action, add_hit_point])

                self.stop_trade(pip, action, position_size)

            if is_stop:
                reward = np.round((hit_point - 20) / 10, 3)
                if days >= self.sim_limit or hit_point == 0:
                    done = 0
                    if not train:
                        _, _, days = self.reset_trade()

                if self.action_type == 'discrete':
                    yield now_state[0], now_state[1], policy, reward, done
                else:
                    yield now_state[0], now_state[1], policy, reward, done
