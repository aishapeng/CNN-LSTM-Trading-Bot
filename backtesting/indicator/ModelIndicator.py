from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar

from agent import CustomAgent
from tensorflow.keras.optimizers import Adam
from collections import deque

import json
import pandas as pd


class ModelIndicator(Indicator):

    def __init__(self,
                 numBars: int,
                 numPrediction: int,
                 intervalToken: str):
        super().__init__([numBars, numPrediction, intervalToken])

        with open("models/2024_10_15_12_22/" + "/Parameters.json", "r") as json_file:
            params = json.load(json_file)
        params["Actor name"] = f"latest_Actor.keras"
        params["Critic name"] = f"latet_Critic.keras"

        self.model =  CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"])

        self.numPrediction = numPrediction
        self.intervalToken = intervalToken
        self.candleData = deque(maxlen=numBars + 1)
        self.numBars = numBars
        # print("DONE")

    def handle_bar(self, bar: Bar):
        self.candleData.append(dict(
            Timestamp=bar.ts_event,
            Open=bar.open.as_double(),
            High=bar.high.as_double(),
            Low=bar.low.as_double(),
            Close=bar.close.as_double(),
        ))

        if len(self.candleData) > self.numBars:
            # data = pd.DataFrame(self.candleData)
            # self.prediction = self.predict(data)
            self._set_initialized(True)
        self.prediction = None

    def isRising(self, col):
        if self.prediction is None:
            self.prediction = self.predict()
        return (self.prediction[col] > 0).all()
    
    def isFalling(self, col):
        if self.prediction is None:
            self.prediction = self.predict()
        return (self.prediction[col] < 0).all()

    def predict(self):
        data = pd.DataFrame(self.candleData)
        action, prediction = self.model.act(data)
        return action
       
    def _reset(self):
        self.candleData.clear()
