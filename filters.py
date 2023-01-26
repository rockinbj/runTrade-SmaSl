import logging

import pandas as pd
from logger import *
from functions import resampleKlines

logger = logging.getLogger("app.filt")


# def f_sma(df, params):
#     # level: sma level
#     # period: sma period
#     # gt==True: close > sma
#     # gt==False: close < sma
#     level, period, gt = params["level"], params["period"], params["gt"]
#     df = df.copy()
#     df = resampleKlines(df, level)
#     df["sma"] = df["close"].rolling(period).mean()
#     df.sort_values("candle_begin_time", inplace=True)
#
#     if pd.isnull(df.iloc[-1]["sma"]):
#         logger.warning(f"f_sma['sma'] is null(nan)")
#     if gt is True:
#         r = df.iloc[-1]["close"] > df.iloc[-1]["sma"]
#     else:
#         r = df.iloc[-1]["close"] < df.iloc[-1]["sma"]
#
#     return r
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


# def f_bias(df, params):
#     # level: bias ma level
#     # period: bias ma length
#     # base: base value for comparing
#     # gt==True: value > base
#     # gt==False: value < base
#     level, period, base, gt = params["level"], params["period"], params["base"], params["gt"]
#     df = df.copy()
#     df = resampleKlines(df, level)
#     df["sma"] = df["close"].rolling(period).mean()
#     df["bias"] = df["close"] / df["sma"] - 1
#     df.sort_values("candle_begin_time", inplace=True)
#
#     if pd.isnull(df.iloc[-1]["bias"]):
#         logger.warning(f"f_bias[{'bias'}] is null(nan)")
#     if gt is True:
#         r = df.iloc[-1]["bias"] > base
#     else:
#         r = df.iloc[-1]["bias"] < base
#
#     return r
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