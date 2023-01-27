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
    df1 = resampleKlines(df, level)
    df1["sma"] = df1["close"].rolling(period).mean()
    # 3位小数，相当于必须>0.01%才有数据，否则就是0
    df1["bias"] = round(df1["close"] / df1["sma"] - 1,4)
    df1.sort_values("candle_begin_time", inplace=True)
    df.sort_values("candle_begin_time", inplace=True)
    df.set_index("candle_begin_time", drop=True, inplace=True)
    df1.set_index("candle_begin_time", drop=True, inplace=True)
    df["bias"] = df1["bias"]
    df["bias"].fillna(method="ffill", inplace=True)

    if gt is True:
        df = df.loc[df["bias"] > base]
    else:
        df = df.loc[df["bias"] < base]

    return df
