from nautilus_trader.model.data import BarType, Bar

from nautilus_trader.config import StrategyConfig

from backtesting.indicator import ModelIndicator
from backtesting.strategy import SimpleStrategy

from data.taskV2.TokenBank import HOUR_1

from nautilus_trader.indicators.average.ama import AdaptiveMovingAverage
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.model.enums import OrderSide, PriceType
from nautilus_trader.model.orders import MarketOrder

import torch

class ModelStrategyConfig(StrategyConfig):
    bar_type: BarType
    stake_size: float

    num_bars: int = 5
    num_predictions: int = 1
    interval_token = HOUR_1
    
    fast_ma: int = 10
    efficiency_ratio_length: int = 10

    ama_fast_ma: int = 21
    ama_slow_ma: int = 49
    diff_threshold: float = 0.01


class ModelStrategy(SimpleStrategy):

    def __init__(self, config: ModelStrategyConfig) -> None:
        super().__init__(config)
        self.bar_type = config.bar_type
        self.instrument_id = config.bar_type.instrument_id
        self.modelIndicator = ModelIndicator(numBars=config.num_bars, 
                                             numPrediction=config.num_predictions,
                                             intervalToken=config.interval_token)
        self.ama = AdaptiveMovingAverage(config.efficiency_ratio_length, 
                                         config.ama_fast_ma, 
                                         config.ama_slow_ma, 
                                         PriceType.LAST)
        self.fast_ma = ExponentialMovingAverage(config.fast_ma)
        self.stake_size = config.stake_size
        self.netShort = self.netLong = False
        self.threshold = config.diff_threshold
        self.lastClose = None
    
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)

        self.register_indicator_for_bars(self.bar_type, self.modelIndicator)
        self.register_indicator_for_bars(self.bar_type, self.ama)
        self.register_indicator_for_bars(self.bar_type, self.fast_ma)

        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized():
            return
        if bar.is_single_price():
            return
        
        signal = self.signal()
        if signal == "buy" and not self.netLong:
            if self.netShort:
                self.close_all_positions(self.instrument_id)
                self.netShort = False
            self.buy()
            self.netLong = True
        elif signal == "sell" and not self.netShort:
            if self.netLong:
                self.close_all_positions(self.instrument_id)
                self.netLong = False
            self.sell()
            self.netShort = True

        
    def signal(self):
        diff = (self.fast_ma.value - self.ama.value) / self.ama.value
        if(self.modelIndicator.isRising("NormHigh") 
           and self.modelIndicator.isRising("NormLow") 
           and diff > self.threshold
        #    and self.modelIndicator.isRising("NormClose")  
        #    and self.modelIndicator.isRising("NormOpen")
           ):
            return "buy"
        elif (self.modelIndicator.isFalling("NormHigh") 
           and self.modelIndicator.isFalling("NormLow") 
           and diff < -self.threshold
        #    and self.modelIndicator.isFalling("NormClose")  
        #    and self.modelIndicator.isFalling("NormOpen")
           ):
            return "sell"
        return None
    
    def pnl(self, close):           
        if self.netLong:
            return (close - self.lastClose) / self.lastClose
        if self.netShort:
            return (self.lastClose - close) / self.lastClose
        return 0
    
    def buy(self) -> None:
        """
        Users simple buy method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.stake_size),
            quote_quantity=True,
            # time_in_force=TimeInForce.FOK,
        )

        self.submit_order(order)

    def sell(self) -> None:
        """
        Users simple sell method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.stake_size),
            quote_quantity=True,
            # time_in_force=TimeInForce.FOK,
        )

        self.submit_order(order)
    
