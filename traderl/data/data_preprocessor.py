import pandas_datareader as pdr
import pandas as pd
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
    """
    Load data from a CSV file or a remote source.

    Parameters:
    remote (bool): If True, load data from a remote source. If False, load data from a local CSV file.
    symbol (str, optional): The symbol of the data to load from the remote source. Required if remote is True.

    Returns:
    pd.DataFrame: The loaded data.
    """
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
    """
    Preprocess data.

    Parameters:
    data (pd.DataFrame): The data to preprocess.
    symbol (str): The symbol of the currency pair.
    financial_type (str): The type of financial data (e.g., 'FX', 'Stock').

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    logging.debug(f"Preprocessing data. Symbol: {symbol}, Financial type: {financial_type}")

    # Convert column names to lowercase
    data.columns = data.columns.str.lower()

    # Determine the pip scale based on the currency pair and financial type
    if financial_type == 'FX':
        pip_scale = 100000 if 'JPY' not in symbol else 1000
        data = data * pip_scale

    # Add technical indicators
    data = add_technical_indicators(data)

    logging.debug(f"Data shape after preprocessing: {data.shape}")
    return data


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the data.

    Parameters:
    data (pd.DataFrame): The data to add the technical indicators to.

    Returns:
    pd.DataFrame: The data with the added technical indicators.
    """
    logging.debug("Adding technical indicators.")

    # Relative Strength Index
    data['rsi'] = ta.momentum.rsi(data['close'])

    # Moving Average Convergence Divergence
    data['macd'] = ta.trend.MACD(data.close).macd_diff()

    # Short-term signal: Crossover of 5-day and 10-day EMA (Exponential Moving Average)
    data['ema_5'] = ta.trend.ema_indicator(data['close'], window=5)
    data['ema_10'] = ta.trend.ema_indicator(data['close'], window=10)
    data['ema_5_10_crossover'] = np.where(data['ema_5'] > data['ema_10'], 1, -1)

    # Long-term signal: 200-day EMA
    data['ema_200'] = ta.trend.ema_indicator(data['close'], window=200)

    # Bollinger Bands
    BollingerBands = ta.volatility.BollingerBands(data['close'])
    data['b_pband'] = BollingerBands.bollinger_pband()
    data['b_wband'] = BollingerBands._hband - BollingerBands._lband

    # Average True Range
    data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])

    # open position
    data['open_position'] = 0

    # close position
    data['close_position'] = 0

    # Drop rows with NaN values
    data = data.dropna()

    logging.debug(f"Data shape after adding technical indicators: {data.shape}")
    return data


def scale_and_split_data(data: pd.DataFrame, window_size: int = 30) -> tuple[pd.DataFrame, np.array]:
    """
    Scale and split the data into chunks of a specified window size.

    Parameters:
    data (pd.DataFrame): The data to scale and split.
    window_size (int, optional): The size of the window to split the data into. Default is 30.

    Returns:
    np.array: The scaled and split data.
    """
    logging.debug(f"Scaling and splitting data. Window size: {window_size}")

    # Select specific columns for the scaled data
    data_array = np.array(data[["rsi", "macd", "b_pband", "b_wband"]])

    # Scale the data
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data_array)

    # Split the data into chunks of the specified window size
    data_split = []
    for i in range(window_size, len(data_scaled)):
        data_split.append(data_scaled[i - window_size:i])
    data = data[window_size:]

    data_split = np.array(data_split)
    data_split = np.transpose(data_split, (0, 2, 1))

    logging.debug(f"Data shape after scaling and splitting: {data_split.shape}")

    return data, data_split


def main(remote: bool, folder_path: str = None, symbols: list = None, financial_type: str = None):
    """
    Main function.

    Parameters:
    remote (bool): If True, load data from a remote source. If False, load data from a local CSV file.
    folder_path (str, optional): The path to the folder containing the CSV files.
    symbols (list, optional): The symbols of the currency pairs.
    financial_type (str, optional): The type of financial data (e.g., 'FX', 'Stock').

    Returns:
    None
    """
    logging.debug(
        f"Running main function. Remote: {remote}, Folder path: {folder_path}, Symbols: {symbols}, Financial type: {financial_type}")

    market_datas = []
    state_datas = []

    # If not remote, get all csv files in the folder path
    if not remote:
        symbols = glob.glob(folder_path + '/*.csv')
        logging.debug(f"Symbols: {symbols}")

    # Load and preprocess data for each symbol
    for symbol in symbols:
        data = load_data(remote, symbol=symbol)
        data = preprocess_data(data, symbol, financial_type)
        data, data_split = scale_and_split_data(data)

        market_datas.append(data)
        state_datas.append(data_split)
    logging.debug(f"Number of items in state_datas: {len(state_datas)}")
    logging.debug(f"Shapes of items in state_datas: {[item.shape for item in state_datas]}")

    # Stack state data
    state_datas = np.array(state_datas)
    logging.debug(f"State data shape: {state_datas.shape}")

    # Split market data into OHLC and stack
    opens = np.stack([data['open'].values for data in market_datas], axis=0)
    highs = np.stack([data['high'].values for data in market_datas], axis=0)
    lows = np.stack([data['low'].values for data in market_datas], axis=0)
    closes = np.stack([data['close'].values for data in market_datas], axis=0)
    atrs = np.stack([data['atr'].values * 2 for data in market_datas], axis=0)

    # Save state data and OHLC data in .npz format
    logging.info("Saving data.")
    np.savez('data.npz', state=state_datas, open=opens, high=highs, low=lows, close=closes, atr=atrs)
    logging.info("Data saved.")

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
