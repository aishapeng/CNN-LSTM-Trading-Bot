import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import rsi
from ta.volume import OnBalanceVolumeIndicator
from utils import Plot_OHCL
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    # df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    # # df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    # # df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    #
    # # Add Bollinger Bands indicator
    # indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    # df['bb_bbm'] = indicator_bb.bollinger_mavg()
    # df['bb_bbh'] = indicator_bb.bollinger_hband()
    # df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator (Trend)
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2,
                                   fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Relative Strength Index (RSI) indicator (Momentum)
    df["rsi"] = rsi(close=df["Close"], window=14, fillna=True)

    # Add Average True Range indicator (Volatility)
    df["atr"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14,
                                 fillna=True).average_true_range()

    # Add On-balance Volume indicator (Volume)
    df["obv"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"], fillna=True).on_balance_volume()


    return df


def DropCorrelatedFeatures(df, threshold, plot):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i + 1, df_corr.shape[0]):
            if df_corr.iloc[i, j] >= threshold or df_corr.iloc[i, j] <= -threshold:
                if columns[j]:
                    columns[j] = False

    selected_columns = df_drop.columns[columns]

    df_dropped = df_drop[selected_columns]

    if plot:
        # Plot Heatmap Correlation
        fig = plt.figure(figsize=(8, 8))
        ax = sns.heatmap(df_dropped.corr(), annot=True, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.show()

    return df_dropped


def get_trend_indicators(df, threshold=0.5, plot=False):
    df_trend = df.copy()

    # add custom trend indicators
    df_trend["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df_trend["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df_trend["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()

    df_trend = add_trend_ta(df_trend, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_trend, threshold, plot)


def get_volatility_indicators(df, threshold=0.5, plot=False):
    df_volatility = df.copy()

    # add custom volatility indicators
    # ...

    df_volatility = add_volatility_ta(df_volatility, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_volatility, threshold, plot)


def get_volume_indicators(df, threshold=0.5, plot=False):
    df_volume = df.copy()

    # add custom volume indicators
    # ...

    df_volume = add_volume_ta(df_volume, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_volume, threshold, plot)


def get_momentum_indicators(df, threshold=0.5, plot=False):
    df_momentum = df.copy()

    # add custom momentum indicators
    # ...

    df_momentum = add_momentum_ta(df_momentum, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_momentum, threshold, plot)


def get_others_indicators(df, threshold=0.5, plot=False):
    df_others = df.copy()

    # add custom indicators
    # ...

    df_others = add_others_ta(df_others, close="Close")

    return DropCorrelatedFeatures(df_others, threshold, plot)


def get_all_indicators(df, threshold=0.5, plot=False):
    df_all = df.copy()

    # add custom indicators
    # ...

    df_all = add_all_ta_features(df_all, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_all, threshold, plot)


def indicators_dataframe(df, threshold=0.5, plot=False):
    trend = get_trend_indicators(df, threshold=threshold, plot=plot)
    volatility = get_volatility_indicators(df, threshold=threshold, plot=plot)
    volume = get_volume_indicators(df, threshold=threshold, plot=plot)
    momentum = get_momentum_indicators(df, threshold=threshold, plot=plot)
    others = get_others_indicators(df, threshold=threshold, plot=plot)
    # all_ind = get_all_indicators(df, threshold=threshold)

    final_df = [df, trend, volatility, volume, momentum, others]
    result = pd.concat(final_df, axis=1)

    return result
