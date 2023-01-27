import logging

import pandas as pd
import pandas_ta as ta
from logger import *
from functions import resampleKlines, getKlineForSymbol

logger = logging.getLogger("app.filt")


def f_sma(df, params):
    # level: sma level
    # period: sma period
    # gt==True: close > sma
    # gt==False: close < sma
    level, period, gt = params["level"], params["period"], params["gt"]
    df.sort_values("candle_begin_time", inplace=True)
    df = df.copy()
    df1 = resampleKlines(df, level)
    df1["sma"] = round(df1["close"].rolling(period).mean(),4)
    df1.sort_values("candle_begin_time", inplace=True)
    df.sort_values("candle_begin_time", inplace=True)
    df.set_index("candle_begin_time", drop=True, inplace=True)
    df1.set_index("candle_begin_time", drop=True, inplace=True)
    df["sma"] = df1["sma"]
    df["sma"].fillna(method="ffill", inplace=True)

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
    df.sort_values("candle_begin_time", inplace=True)
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
    df.sort_values("candle_begin_time", inplace=True)
    df = df.copy()
    df1 = resampleKlines(df, level)
    df1["sma"] = df1["close"].rolling(period).mean()
    # 3位小数，相当于必须>0.01%才有数据，否则就是0
    df1["bias"] = round(df1["close"] / df1["sma"] - 1, 4)
    df1.sort_values("candle_begin_time", inplace=True)
    df.sort_values("candle_begin_time", inplace=True)
    df.set_index("candle_begin_time", drop=True, inplace=True)
    df1.set_index("candle_begin_time", drop=True, inplace=True)
    df["sma"] = df1["sma"]
    df["bias"] = df1["bias"]
    df["bias"].fillna(method="ffill", inplace=True)

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
    df.sort_values("candle_begin_time", inplace=True)
    df = df.copy()
    level, periods, long = params["level"], params["periods"], params["long"]
    df1 = resampleKlines(df, level)
    p1 = "ema" + str(periods[0])
    p2 = "ema" + str(periods[1])
    p3 = "ema" + str(periods[2])
    df1.sort_values("candle_begin_time", inplace=True)
    df1[p1] = ta.ema(df1["close"], length=periods[0])
    df1[p2] = ta.ema(df1["close"], length=periods[1])
    df1[p3] = ta.ema(df1["close"], length=periods[2])

    df.sort_values("candle_begin_time", inplace=True)
    df.set_index("candle_begin_time", drop=True, inplace=True)
    df1.set_index("candle_begin_time", drop=True, inplace=True)
    df[p1] = df1[p1]
    df[p2] = df1[p2]
    df[p3] = df1[p3]
    df[[p1, p2, p3]] = df[[p1, p2, p3]].fillna(method="ffill")

    if long is True:
        df = df.loc[(df[p1] > df[p2]) & (df[p2] > df[p3])]
    else:
        df = df.loc[(df[p1] < df[p2]) & (df[p2] < df[p3])]

    return df
