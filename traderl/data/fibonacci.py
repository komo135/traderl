"""
Generate Fibonacci retracement data.
"""
import logging
import json

import pandas as pd
import numpy as np
import ta


class FibonacciRetracementGenerator:
    """
    Generate Fibonacci retracement data.
    """

    def __init__(self, data: pd.DataFrame, ma_windows: tuple[int, int] = (50, 100)) -> None:
        self.data = data
        self.__calc_moving_average(ma_windows)
        self.__detect_trend_cross()

    def __calc_moving_average(self, ma_windows: tuple[int, int]) -> pd.DataFrame:
        """
        Calculate moving average.
        """
        if ma_windows[0] >= ma_windows[1]:
            raise ValueError("Short MA window must be less than long MA window.")

        logging.info("Calculating moving average...")
        self.short_ma = ta.trend.ema_indicator(self.data, window=ma_windows[0])
        self.long_ma = ta.trend.ema_indicator(self.data, window=ma_windows[1])
        logging.info("Moving average calculated.")

    def __detect_trend_cross(self) -> None:
        """
        Detect trend cross with confirmation.
        """
        logging.info("Detecting trend cross...")
        self.data["trend"] = np.where(self.short_ma > self.long_ma, 1, -1)
        self.data["trend_shift"] = self.data["trend"].shift(1)
        self.data["trend_cross"] = np.where(self.data["trend"] != self.data["trend_shift"], True, False)

        # 確認期間を設定
        confirmation_period = 10
        self.data["confirmed_trend"] = self.data["trend"]

        # ノイズを除去
        for i in range(confirmation_period, len(self.data)):
            if self.data["trend_cross"].iloc[i]:
                if all(self.data["trend"].iloc[i:i+confirmation_period] == self.data["trend"].iloc[i]):
                    self.data["confirmed_trend"].iloc[i] = self.data["trend"].iloc[i]
                else:
                    self.data["confirmed_trend"].iloc[i] = self.data["confirmed_trend"].iloc[i-1]
            else:
                self.data["confirmed_trend"].iloc[i] = self.data["confirmed_trend"].iloc[i-1]

        logging.info("Trend cross detected with confirmation.")

        self.data = self.data[["open", "high", "low", "close", "confirmed_trend"]]
        self.data = self.data.dropna()

    def __detect_high_low(self, start_index: int, now_index: int) -> tuple[int, int]:
        """
        Detect high low.

        Args:
            start_index (int): Start index.
            now_index (int): Now index.

        Returns:
            tuple[int, int]: High price, low price.
        """
        logging.info("Detecting high low...")

        now_index += 1
        start_index -= 100  # トレンドが遅れていることを考慮

        high_price = self.data["high"].iloc[start_index:now_index].max()
        low_price = self.data["low"].iloc[start_index:now_index].min()

        return high_price, low_price

    def __calc_fibonacci_retracement(self, high_price: float, low_price: float, trend: int) -> tuple[float, dict]:
        """
        Calculate the percentage level of the current price within the Fibonacci retracement levels
        and determine the current retracement level.

        Args:
            high_price (float): High price.
            low_price (float): Low price.
            trend (int): Current trend direction (1 for uptrend, -1 for downtrend).

        Returns:
            tuple[float, dict]: Diff and Fibonacci retracement levels and current retracement level.
        """
        diff = high_price - low_price
        if diff == 0:
            return {"retracement_level": 0.0, "current_level": None}  # Avoid division by zero

        # トレンドに基づいてフィボナッシィレトレンドのレベルを計算
        if trend == 1:  # Uptrend
            fib_levels = {
                "23.6%": high_price - 0.236 * diff,
                "38.2%": high_price - 0.382 * diff,
                "50.0%": high_price - 0.5 * diff,
                "61.8%": high_price - 0.618 * diff,
                "78.6%": high_price - 0.786 * diff
            }
        else:  # Downtrend
            fib_levels = {
                "23.6%": low_price + 0.236 * diff,
                "38.2%": low_price + 0.382 * diff,
                "50.0%": low_price + 0.5 * diff,
                "61.8%": low_price + 0.618 * diff,
                "78.6%": low_price + 0.786 * diff
            }

        return diff, fib_levels

    def __calc_fibonacci_retracement_level(self, now_price: float, trend: int, fib_data: dict) -> tuple[float, float]:
        """
        Calculate Fibonacci retracement level.

        Args:
            now_price (float): Current price.
            trend (int): Current trend direction (1 for uptrend, -1 for downtrend).
            fib_data (dict): Dictionary containing high_price, low_price, diff, and fib_levels.

        Returns:
            tuple[float, float]: Retracement level and current retracement level.
        """
        high_price = fib_data["high_price"]
        low_price = fib_data["low_price"]
        diff = fib_data["diff"]
        fib_levels = fib_data["fib_levels"]

        if trend == 1:  # Uptrend
            retracement_level = (high_price - now_price) / diff
        else:  # Downtrend
            retracement_level = (now_price - low_price) / diff

        current_level = None
        for level, price in fib_levels.items():
            if (trend == 1 and now_price >= price) or (trend == -1 and now_price <= price):
                current_level = float(level.replace("%", "")) / 100

        return retracement_level, current_level

    def calculate(self) -> None:
        """
        Calculate Fibonacci retracement.
        """
        logging.info("Calculating Fibonacci retracement...")
        self.data["fib_retracement_level"] = np.nan
        self.data["fib_retracement_current_level"] = np.nan

        fib_data = {
            "high_price": None,
            "low_price": None,
            "diff": None,
            "fib_levels": {}
        }

        trend = self.data["confirmed_trend"][0]
        is_trend_change = False
        start_index = 0

        for i in range(len(self.data["close"])):
            now_price = self.data["close"].iloc[i]

            if trend != self.data["confirmed_trend"][i]:
                logging.info("Trend changed.")
                logging.debug("trend: %d, now_price: %f", trend, now_price)

                trend = self.data["confirmed_trend"][i]
                is_trend_change = True

                start_index = i

            if start_index >= 100 and (is_trend_change or now_price > fib_data["high_price"] or now_price < fib_data["low_price"]):
                is_trend_change = False

                fib_data["high_price"], fib_data["low_price"] = self.__detect_high_low(start_index, i)
                fib_data["diff"], fib_data["fib_levels"] = self.__calc_fibonacci_retracement(fib_data["high_price"], fib_data["low_price"], trend)

                logging.debug("fib_data: \n%s", json.dumps(fib_data, indent=4))

            if fib_data["diff"] is not None:
                retracement_level, current_level = self.__calc_fibonacci_retracement_level(now_price, trend, fib_data)
                self.data["fib_retracement_level"].iloc[i] = retracement_level
                self.data["fib_retracement_current_level"].iloc[i] = current_level
