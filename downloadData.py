from binance.client import Client
import pandas as pd
import datetime

# Replace these with your own Binance API key and secret
api_key = 'xueN0LzUIX9zRV6KvhMe7cv4hFj0XnO3nK9BVNta62eg3WGXH8B1wELphc0Gl4mt'
api_secret = 'PlyL3kuKiQfMp40Vq50wNrlFy7xbuepZQ3Xri23oMVpomltUhdLZT65LV6gnhLeS'

# Initialize the Binance Client
client = Client(api_key, api_secret)

def get_binance_btc_data(symbol, interval, start_str, end_str=None):
    """
    Fetches historical data for the given symbol and interval from Binance.

    :param symbol: The trading symbol (e.g. 'BTCUSDT').
    :param interval: The data interval (e.g. '1h' for 1-hour intervals).
    :param start_str: The start date as a string (e.g. '1 Jan, 2021').
    :param end_str: The end date as a string, default is None (it will fetch data till the current time).
    :return: A DataFrame containing the historical data.
    """
    # Fetch the klines (OHLCV data) from Binance
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    # Convert to DataFrame with specific column names
    df = pd.DataFrame(klines, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])

    # Convert timestamps to readable date format (optional)
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # Convert numeric columns from strings to floats
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Drop unnecessary columns and keep only the required ones
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    return df

# Example: Fetch 1-hour interval BTC data starting from January 1, 2023
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR  # '1h' for 1-hour interval
start_date = '1 Apr, 2024'
end_date = '4 Oct, 2024'

btc_data = get_binance_btc_data(symbol, interval, start_date, end_date)

# Save the data to a CSV file
btc_data.to_csv('btc_1h_data_testing.csv', index=False)

# Print the first few rows of the data
print(btc_data.head())
