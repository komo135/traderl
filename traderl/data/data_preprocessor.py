import pandas_datareader as pdr
import pandas as pd
from datetime import datetime
import glob
import numpy as np
import ta
from sklearn.preprocessing import RobustScaler
import argparse
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)


def load_data(remote: bool, symbol: str = None) -> pd.DataFrame:
    start_time = time.time()
    logging.debug(f"Loading data. Remote: {remote}, Symbol: {symbol}")
    if remote:
        if symbol is None:
            raise ValueError("Symbol are required when loading data from a remote source.")
        data = pdr.get_data_yahoo(symbol)
    else:
        file_path = symbol
        if file_path is None:
            raise ValueError("File path is required when loading data from a local file.")
        data = pd.read_csv(file_path)
    logging.debug(f"Data loaded in {time.time() - start_time} seconds.")
    return data


def preprocess_data(data: pd.DataFrame, symbol: str, financial_type: str) -> pd.DataFrame:
    logging.debug(f"Preprocessing data. Symbol: {symbol}, Financial type: {financial_type}")
    data.columns = data.columns.str.lower()
    if financial_type == 'FX':
        pip_scale = 100000 if 'JPY' not in symbol else 1000
        data = data * pip_scale
    data = add_technical_indicators(data)
    logging.debug(f"Data shape after preprocessing: {data.shape}")
    return data


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    logging.debug("Adding technical indicators.")
    data['rsi'] = ta.momentum.rsi(data['Close'])
    data['macd'] = ta.trend.MACD(data.close).macd_diff()
    data['ema_5'] = ta.trend.ema_indicator(data['Close'], window=5)
    data['ema_10'] = ta.trend.ema_indicator(data['Close'], window=10)
    data['ema_5_10_crossover'] = np.where(data['ema_5'] > data['ema_10'], 1, -1)
    data['ema_200'] = ta.trend.ema_indicator(data['Close'], window=200)
    BollingerBands = ta.volatility.BollingerBands(data['Close'])
    data['b_pband'] = BollingerBands.bollinger_pband()
    data['b_wband'] = BollingerBands.bollinger_wband()
    data['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    logging.debug(f"Data shape after adding technical indicators: {data.shape}")
    return data


def scale_and_split_data(data: pd.DataFrame, window_size: int = 30) -> np.array:
    logging.debug(f"Scaling and splitting data. Window size: {window_size}")
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = np.array(data_scaled[["ema_5_10_crossover", "ema_200", "rsi", "macd", "b_pband", "b_wband", "atr"]])
    data_split = []
    for i in range(len(data_scaled) - window_size + 1):
        data_split.append(data_scaled[i:i + window_size])
    logging.debug(f"Data shape after scaling and splitting: {data.shape}")
    return np.array(data_split)


def main(remote: bool, folder_path: str = None, symbols: list = None, financial_type: str = None):
    logging.debug(
        f"Running main function. Remote: {remote}, Folder path: {folder_path}, Symbols: {symbols}, Financial type: {financial_type}")
    market_datas = []
    state_datas = []
    if not remote:
        symbols = glob.glob(folder_path + '/*.csv')
    for symbol in symbols:
        data = load_data(remote, symbol=symbol)
        data = preprocess_data(data, symbol, financial_type)
        market_datas.append(data)
        state_datas.append(scale_and_split_data(data))
    state_datas = np.stack(state_datas, axis=0)
    opens = np.stack([data['open'].values for data in market_datas], axis=0)
    highs = np.stack([data['high'].values for data in market_datas], axis=0)
    lows = np.stack([data['low'].values for data in market_datas], axis=0)
    closes = np.stack([data['close'].values for data in market_datas], axis=0)
    atrs = np.stack([data['atr'].values for data in market_datas], axis=0)
    np.savez('data.npz', state=state_datas, open=opens, high=highs, low=lows, close=closes, atr=atrs)
    logging.debug("Main function completed.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--remote', action='store_true', help='Load data from a remote source.')
    arg_parser.add_argument('--folder_path', type=str, default='historical_data',
                            help='The path to the folder containing the CSV files.')
    arg_parser.add_argument('--symbols', nargs='+', help='The symbols of the currency pairs.')
    arg_parser.add_argument('--financial_type', type=str, help='The type of financial data (e.g., FX, Stock).')
    args = arg_parser.parse_args()

    main(args.remote, args.folder_path, args.symbols, args.financial_type)
