import random
import pandas as pd

from logger import *

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
logger = logging.getLogger("app.sig")


# 所有signal都要输出一个带有chosen列的df，是选出的交易币种
# 传递给getPosition()，作为下一周期持仓依据

# 计算周期内的涨幅
# 例如20日涨幅
def s_mtm(df, para=20):
    n = para
    df = df.copy()
    return df["close"].pct_change(periods=n)


def s_bias(df, para=20):
    n = para
    df = df.copy()
    df["ma"] = df["close"].rolling(n).mean()
    df["bias"] = df["close"] / df["ma"] - 1
    return df["bias"]


def s_sma(df, para=20):
    n = para
    df = df.copy()
    return df["close"].rolling(n).mean()


def s_ema(df, para=20):
    n = para
    df = df.copy()
    name = "ema" + str(n)
    df[name] = pd.DataFrame.ewm(df["close"], span=n).mean()
    return df[name]


def s_test(df, para):
    name = "signalTest"
    n = para
    df = df.copy()
    df[name] = random.random() * -1

    return df[name]
