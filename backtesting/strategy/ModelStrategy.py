from nautilus_trader.model.data import BarType, Bar

from nautilus_trader.config import StrategyConfig

from backtesting.indicator import ModelIndicator
from backtesting.strategy import SimpleStrategy

from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.atr import AverageTrueRange

from nautilus_trader.indicators.average.ama import AdaptiveMovingAverage
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.model.enums import OrderSide, PriceType
from nautilus_trader.model.orders import MarketOrder
from ta.volume import ChaikinMoneyFlowIndicator


class ModelStrategyConfig(StrategyConfig):
    bar_type: BarType
    stake_size: float

    num_bars: int = 5
    num_predictions: int = 1
    
    fast_ma: int = 7
    rsi_length: int = 7
    atr_length: int = 7
    efficiency_ratio_length: int = 10

    ama_fast_ma: int = 5
    ama_slow_ma: int = 10
    diff_threshold: float = 0.01


class ModelStrategy(SimpleStrategy):

    def __init__(self, config: ModelStrategyConfig) -> None:
        super().__init__(config)
        self.bar_type = config.bar_type
        self.instrument_id = config.bar_type.instrument_id
        self.modelIndicator = ModelIndicator(numBars=config.num_bars, 
                                             numPrediction=config.num_predictions,
                                             )
        self.rsi = RelativeStrengthIndex(config.rsi_length)
        self.prev_rsi = 0
        # self.atr = AverageTrueRange(config.atr_length)
        self.ama = AdaptiveMovingAverage(config.efficiency_ratio_length, 
                                         config.ama_fast_ma, 
                                         config.ama_slow_ma, 
                                         PriceType.LAST)
        self.prev_ama = 0
        self.fast_ma = ExponentialMovingAverage(config.fast_ma)
        self.low_streak = 0
        self.stake_size = config.stake_size
        self.netShort = self.netLong = False
        self.threshold = config.diff_threshold
        self.lastClose = None
        self.threshold = config.diff_threshold
    
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        self.register_indicator_for_bars(self.bar_type, self.modelIndicator)
        self.register_indicator_for_bars(self.bar_type, self.rsi)
        # self.register_indicator_for_bars(self.bar_type, self.atr)
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
        
        self.prev_rsi = self.rsi.value
        self.prev_ema = self.fast_ma.value

        
    def signal(self):
        # diff = (self.fast_ma.value - self.ama.value) / self.ama.value
        if (self.modelIndicator.predict() == 1 and self.rsi.value<0.7 and self.low_streak<=3):
            return "buy"
        elif (self.modelIndicator.predict() == 2):
            return "sell"
        return None
    
    def check_lower_low(self, current_value):
        if self.fast_ma.value < self.prev_ema:
            self.low_streak += 1
        else:
            self.low_streak = 0
    
    def check_crossover(self):
        if self.prev_rsi < self.prev_ema and self.rsi.value > self.fast_ma.value:
            return "buy"
        elif self.prev_rsi > self.prev_ema and self.rsi.value < self.fast_ma.value:
            return "sell"
        else:
            return "no"
    
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
    
