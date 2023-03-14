import random
import pandas as pd
import pandas_ta as ta

from utils.logger import *

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
logger = logging.getLogger("app.sig")


# 计算周期内的涨幅
# 例如20日涨幅
def s_mtm(df, para=20):
    df.sort_values("candle_begin_time", inplace=True)
    n = para
    df = df.copy()
    return df["close"].pct_change(periods=n)


def s_bias(df, para=20):
    df.sort_values("candle_begin_time", inplace=True)
    n = para
    df = df.copy()
    df["ma"] = df["close"].rolling(n).mean()
    df["bias"] = df["close"] / df["ma"] - 1

    return df["bias"]


def s_sma(df, para=20):
    df.sort_values("candle_begin_time", inplace=True)
    n = para
    df = df.copy()
    return df["close"].rolling(n).mean()


def s_ema(df, para=20):
    df.sort_values("candle_begin_time", inplace=True)
    n = para
    df = df.copy()
    name = "ema" + str(n)
    df[name] = ta.ema(df["close"], length=n)
    return df[name]


def s_test(df, para):
    df.sort_values("candle_begin_time", inplace=True)
    name = "signalTest"
    df = df.copy()
    df[name] = random.random() * -1

    return df[name]
