import logging

import pandas as pd
from logger import *
from functions import resampleKlines

logger = logging.getLogger("app.filt")


def f_sma(df, params):
    # level: sma level
    # period: sma period
    # gt==True: close > sma
    # gt==False: close < sma
    level, period, gt = params["level"], params["period"], params["gt"]
    df = df.copy()
    df = resampleKlines(df, level)
    df["sma"] = df["close"].rolling(period).mean()
    df.sort_values("candle_begin_time", inplace=True)

    if gt is True:
        df = df.loc[df["close"] > df["sma"]]
    else:
        df = df.loc[df["close"] < df["sma"]]

    return df


def f_compare(df, params):
    # _column: column name
    # base: base value for comparing
    # gt==True: value > base
    # gt==False: value < base
    _column, base, gt = params["_column"], params["base"], params["gt"]
    df = df.copy()
    df.sort_values("candle_begin_time", inplace=True)

    if pd.isnull(df.iloc[-1][_column]):
        logger.warning(f"f_compare[{_column}] is null(nan)")
    if gt is True:
        r = df.iloc[-1][_column] > base
    else:
        r = df.iloc[-1][_column] < base

    return r


def f_bias(df, params):
    # level: bias ma level
    # period: bias ma length
    # base: base value for comparing
    # gt==True: value > base
    # gt==False: value < base
    level, period, base, gt = params["level"], params["period"], params["base"], params["gt"]
    df = df.copy()
    df = resampleKlines(df, level)
    df["sma"] = df["close"].rolling(period).mean()
    df["bias"] = df["close"] / df["sma"] - 1
    df.sort_values("candle_begin_time", inplace=True)

    if gt is True:
        df = df.loc[df["bias"] > base]
    else:
        df = df.loc[df["bias"] < base]

    return df


def f_ema_trend(df, params):
    # ema趋势判断，多头排列、空头排列
    # periods = [20,60,120]，必须三个
    # long==True: ema20>ema60>ema120
    # long==False: ema20<ema60<ema120
    df = df.copy()
    periods, long = params["periods"], params["long"]
    p1 = "ema" + str(periods[0])
    p2 = "ema" + str(periods[1])
    p3 = "ema" + str(periods[2])
    df[p1] = pd.DataFrame.ewm(df["close"], span=periods[0]).mean()
    df[p2] = pd.DataFrame.ewm(df["close"], span=periods[1]).mean()
    df[p3] = pd.DataFrame.ewm(df["close"], span=periods[2]).mean()

    if long is True:
        df = df.loc[(df[p1] > df[p2]) & (df[p2] > df[p3])]
    else:
        df = df.loc[(df[p1] < df[p2]) & (df[p2] < df[p3])]

    return df
