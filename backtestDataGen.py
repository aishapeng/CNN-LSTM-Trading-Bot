# Importing Necessary Modules
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.node import (
    BacktestNode,
    BacktestVenueConfig,
    BacktestDataConfig,
    BacktestRunConfig,
    BacktestEngineConfig,
)
from nautilus_trader.core.datetime import dt_to_unix_nanos

from nautilus_trader.persistence.wranglers import (
    QuoteTickDataWrangler,
)

from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.identifiers import Venue, InstrumentId, Symbol
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.currencies import BTC, USDT

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig

import pandas as pd
from decimal import Decimal

from indicators import AddIndicators
from utils import Normalizing

# Initialize the Data Catalog
catalog = ParquetDataCatalog("./backtest_catalog")

# Define the BTC/USDT 1-Hour Trading Pair
btc_usdt_1h = CurrencyPair(
    instrument_id=InstrumentId(
        symbol=Symbol("BTCUSDT"),
        venue=Venue("BINANCE"),
    ),
    raw_symbol=Symbol("BTCUSDT"),
    base_currency=BTC,
    quote_currency=USDT,
    price_precision=2,
    size_precision=6,
    price_increment=Price(1e-02, precision=2),
    size_increment=Quantity(1e-06, precision=6),
    lot_size=None,
    max_quantity=Quantity(1e10, precision=1),
    min_quantity=Quantity(1e-06, precision=6),
    max_notional=None,
    min_notional=Money(10.00000000, USDT),
    max_price=Price(1000000, precision=2),
    min_price=Price(0.001, precision=3),
    margin_init=Decimal(0),
    margin_maint=Decimal(0),
    maker_fee=Decimal("0.001"),
    taker_fee=Decimal("0.001"),
    ts_event=0,
    ts_init=0,
)

# Write Instrument Configuration to the Catalog
catalog.write_data([btc_usdt_1h])

# Path to the Local BTC/USDT 1-Hour CSV Data
DATA_FILE_PATH = "./data/btc_1h_data_testing.csv"

# Load and Process the BTC 1-Hour Data
try:
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(DATA_FILE_PATH)
    print(data.columns)

    # data = data.rename(
    #     columns={'timestamp': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    #              'volume': 'Volume'})

    # test_df = AddIndicators(data)  # insert indicators
    # test_df = test_df[100:] # remove invalid indicators value
    # data = data[100:]
    # data = data[data[:] != 0] # remove 0 to prevent math error from logging

    # test_df = Normalizing(test_df).dropna()
    # data = data[1:] # remove nan from normalizing
    # test_df = test_df[1:]

    # # Rename Columns to Match Expected Format
    # data = data.rename(columns={
    #     "Timestamp": "timestamp",
    #     "Open": "open",
    #     "High": "high",
    #     "Low": "low",
    #     "Close": "close",
    #     "Volume": "volume"
    # })

    # # Ensure the 'timestamp' column is in datetime format (assuming UNIX milliseconds)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # # Set 'timestamp' as the DataFrame index
    data = data.set_index("timestamp")

    # # Optional: Sort the DataFrame by timestamp if not already sorted
    data = data.sort_index()

    # Initialize the Data Wrangler for Quote Ticks
    wrangler = QuoteTickDataWrangler(btc_usdt_1h)

    # Process the Bar Data
    # Since the data is already 1-hour bars, set both 'bid' and 'ask' data as the same
    # Adjust 'offset_interval_ms' to 3600000 for 1-hour intervals
    quote_data = wrangler.process_bar_data(
        data,
        data,
        offset_interval_ms=3600000  # 1 hour in milliseconds
    )

    # Write the Processed Quote Data to the Catalog
    catalog.write_data(quote_data)

    print("Data processing and catalog writing completed successfully.")

except FileNotFoundError:
    print(f"Error: The data file at {DATA_FILE_PATH} was not found.")
except pd.errors.EmptyDataError:
    print("Error: The data file is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
