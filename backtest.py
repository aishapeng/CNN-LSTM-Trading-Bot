from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.node import BacktestNode, BacktestVenueConfig, BacktestDataConfig, BacktestRunConfig, BacktestEngineConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos

from nautilus_trader.persistence.wranglers_v2 import BarDataWranglerV2

from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.model.data import BarType, BarSpecification, BarAggregation, Bar, TradeTick, QuoteTick
from nautilus_trader.model.enums import AccountType, PriceType, AggregationSource, OmsType
from nautilus_trader.model.identifiers import Venue, Symbol, InstrumentId
from nautilus_trader.config import ImportableStrategyConfig, LoggingConfig, StrategyConfig
from nautilus_trader.trading.strategy import Strategy

# from datasets import load_dataset
import pandas as pd
from datetime import datetime


# catalog = ParquetDataCatalog("./backtest_catalog")
# print(catalog.instruments())

if __name__ == "__main__":
    symbolList = ["BTCUSDT"]
    symbol = "BTCUSDT"
    # bar_spec = BarSpecification(1, BarAggregation.HOUR, 1)
    instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")
    bar = BarType(instrument_id, BarSpecification(1, BarAggregation.HOUR, 1), AggregationSource.INTERNAL)

    # wrangler = BarDataWrangler(barType, instrument)

    start_date = dt_to_unix_nanos(pd.Timestamp("2024-04-01 01:00:00", tz="UTC"))
    end_date = dt_to_unix_nanos(pd.Timestamp("2024-10-01 00:00:00", tz="UTC"))


    venue_configs = [
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="HEDGING",
            account_type="MARGIN",
            base_currency=None,
            starting_balances=["10_000 USDT"],
        ),
    ]

    data_configs = [
        BacktestDataConfig(
            catalog_path="./backtest_catalog",
            data_cls=QuoteTick,
            instrument_id=instrument_id,
            start_time=start_date,
            end_time=end_date,
        ),
    ]

    strategies = [
        ImportableStrategyConfig(
            strategy_path="backtesting.strategy.ModelStrategy:ModelStrategy",
            config_path="backtesting.strategy.ModelStrategy:ModelStrategyConfig",
            config=dict(
                # instrument_id=instrument_id,
                # bar_spec=bar_spec,
                bar_type = bar,
                stake_size=5_000,
            ),
        ),
    ]

    config = BacktestRunConfig(
        engine=BacktestEngineConfig(
            strategies=strategies, logging=LoggingConfig(log_level="INFO")),
        data=data_configs,
        venues=venue_configs,
        # batch_size_bytes=1024 * 1024
    )
    node = BacktestNode(configs=[config])
    # node = MultiBacktestNode(configs=configList)

    results = node.run()
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")
    node.get_engines()[0].trader.generate_order_fills_report().to_csv(
        f"./backtest_result/{date}_order_report.csv")
    node.get_engines()[0].trader.generate_positions_report().to_csv(
        f"./backtest_result/{date}_position_report.csv")
    node.get_engines()[0].trader.generate_account_report(
        Venue("BINANCE")).to_csv(f"./backtest_result/{date}_account_report.csv")
    print(results)