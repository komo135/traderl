import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traderl.environment.env import Env


class Evolution:
    """
    This class simulates the evolution of the trading environment.
    It allows to reset the environment, simulate the evolution with a given action function, and plot the trade history.

    Example
    -------
    ```python
    import torch
    from traderl.agent import Agent
    from traderl.environment import Environment
    from traderl.agent.common import Evolution

    # Create an agent
    agent = Agent()
    # Create an environment
    env = Environment()
    # Create an evolution
    evolution = Evolution(env)
    # Evolute the environment
    evolution.evolute(agent.get_action, 0, 1000)
    ```
    """

    def __init__(self, env: Env) -> None:
        """
        Initialize the Evolution class.

        Parameters
        ----------
        env : Env
            The environment object.
        """
        self.env = env
        self._reset()

    def _reset(self):
        """
        Reset the environment.
        """
        self.env.symbol = 0
        self.trade_historys = []
        self.total_pips = []

    def _plot(self, symbol: int, start: str, end: str):
        """
        Plot the trade history.

        Parameters
        ----------
        symbol : int
            The index of the symbol.
        start : int
            The start index.
        end : int
            The end index.
        """
        trade_history = self.trade_historys[symbol][start:end]
        data = self.env.data
        open = data['open'][symbol][start:end]
        high = data['high'][symbol][start:end]
        low = data['low'][symbol][start:end]
        close = data['close'][symbol][start:end]
        ohlc = np.stack([open, high, low, close], axis=-1)
        ohlc_df = pd.DataFrame(ohlc, columns=['open', 'high', 'low', 'close'])

        _, ax = plt.subplots()
        mpf.plot(ohlc_df, type='candle', ax=ax)

        for event, marker, color in zip(['long', 'short', 'stop loss', 'take profit', 'stop trade'],
                                        ['^', 'v', 'x', 'o', 's'],
                                        ['g', 'r', 'b', 'y', 'c']):
            indices = [i for i, e in enumerate(trade_history) if e == event]
            ax.plot(indices, ohlc_df['close'][indices], marker, markersize=10, color=color, label=event)

        plt.legend()
        plt.show()

    def evolute(self, get_action, strat, end, plot_symbol=0):
        """
        Simulate the evolution of the environment.

        Parameters
        ----------
        get_action : function
            Function to get the action.
        strat : int
            The start index.
        end : int
            The end index.
        plot_symbol : int, optional
            The index of the symbol to plot, by default 0
        """
        self._reset()
        for _ in range(self.env.symbols):
            step = self.env.step(get_action, strat, end)
            for _ in step:
                pass
            self.trade_historys.append(self.env.trade_history)
            self.total_pips.append(self.env.total_pip)

        print_txt = ", ".join(f"{i}: {total_pip}" for i, total_pip in enumerate(self.total_pips))
        print(print_txt)
        self._plot(plot_symbol, strat, end)


__all__ = ['Evolution']
