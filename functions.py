import json
from functools import partial
from functools import reduce
from multiprocessing import Pool, cpu_count

import ccxt
import pandas as pd
import requests
from tenacity import *

import signals
from exchangeConfig import *
from logger import *
from settings import *


# pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
# pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)
logger = logging.getLogger("app.func")


def retryIt(fun, paras={}, retryTimes=MAX_TRY, retryWaitFor=RETRY_WAIT, critical=False):
    for i in range(retryTimes):
        try:
            r = fun(params=paras)
            return r
        except ccxt.MarginModeAlreadySet:
            pass
        except Exception as e:
            logger.error(f"{fun.__name__}报错, retryIt()即将在{retryWaitFor}秒后重试{i}次: {e}")
            logger.exception(e)
            time.sleep(retryWaitFor)
            if i == retryTimes - 1:
                f = f"{STRATEGY_NAME} {fun.__name__} retryIt()重试{retryTimes}次无效, 请检查日志。程序提交异常。{e}"
                if critical:
                    sendAndCritical("！严重级别告警！" + f)
                else:
                    sendAndPrintError(f)
                raise RuntimeError(f"{fun.__name__} {e}")
            continue


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def callAlarm(strategyName=STRATEGY_NAME, content="存在严重风险项, 请立即检查"):
    url = "http://api.aiops.com/alert/api/event"
    apiKey = "66e6aeab4218431f8afe7e76ac96c38e"
    eventId = str(int(time.time()))
    stragetyName = strategyName
    content = content
    para = f"?app={apiKey}&eventType=trigger&eventId={eventId}&priority=3&host={stragetyName}&alarmContent={content}"

    try:
        r = requests.post(url + para)
        if r.json()["result"] != "success":
            sendAndPrintError(f"电话告警触发失败, 可能有严重风险, 请立即检查！{r.text}")
    except Exception as e:
        logger.error(f"电话告警触发失败, 可能有严重风险, 请立即检查！{e}")
        logger.exception(e)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def sendMixin(msg, _type="PLAIN_TEXT"):
    token = MIXIN_TOKEN
    url = f"https://webhook.exinwork.com/api/send?access_token={token}"

    value = {
        'category': _type,
        'data': msg,
    }

    try:
        r = requests.post(url, data=value, timeout=2).json()
    except Exception as err:
        logger.exception(err)


def sendAndPrintInfo(msg):
    logger.info(msg)
    sendMixin(msg)


def sendAndPrintError(msg):
    logger.error(msg)
    sendMixin(msg)


def sendAndCritical(msg):
    logger.critical(msg)
    if CALL_ALARM:
        callAlarm(strategyName=STRATEGY_NAME, content=msg)
    sendMixin(msg)


def sendAndRaise(msg):
    logger.error(msg)
    sendMixin(msg)
    raise RuntimeError(msg)


def sendReport(exchangeId, interval=REPORT_INTERVAL):
    exchange = getattr(ccxt, exchangeId)(EXCHANGE_CONFIG)

    nowMinute = dt.datetime.now().minute
    nowSecond = dt.datetime.now().second

    if (nowMinute % interval == 0) and (nowSecond == 59):
        logger.debug("开始发送报告")

        pos = getOpenPosition(exchange)
        bTot, bBal, bPos = getBalances(exchange)
        bal = round(float(bTot.iloc[0]["availableBalance"]), 2)
        wal = round(float(bTot.iloc[0]["totalMarginBalance"]), 2)

        msg = f"### {STRATEGY_NAME} - 策略报告\n\n"

        if pos.shape[0] > 0:
            pos = pos[
                [
                    "notional",
                    "percentage",
                    "unrealizedPnl",
                    "entryPrice",
                    "markPrice",
                    "liquidationPrice",
                    "datetime",
                    "leverage",
                ]
            ]
            pos.rename(
                columns={
                    "notional": "持仓价值(U)",
                    "percentage": "盈亏比例(%)",
                    "unrealizedPnl": "未实现盈亏(U)",
                    "entryPrice": "开仓价格(U)",
                    "markPrice": "标记价格(U)",
                    "liquidationPrice": "爆仓价格(U)",
                    "datetime": "开仓时间",
                    "leverage": "杠杆倍数",
                },
                inplace=True,
            )
            pos.sort_values(by="盈亏比例(%)", ascending=False, inplace=True)
            d = pos.to_dict(orient="index")

            msg += f"#### 账户权益 : {wal}U\n"
            msg += f"#### 可用余额 : {bal}U\n"
            msg += f'#### 当前持币 : {", ".join(list(d.keys()))}'

            for k, v in d.items():
                msg += f"""
##### {k}
 - 持仓价值(U) : {v["持仓价值(U)"]}
 - 盈亏比例(%) : {v["盈亏比例(%)"]}
 - 未实现盈亏(U) : {v["未实现盈亏(U)"]}
 - 开仓价格(U) : {v["开仓价格(U)"]}
 - 标记价格(U) : {v["标记价格(U)"]}
 - 爆仓价格(U) : {v["爆仓价格(U)"]}
 - 开仓时间 : {v["开仓时间"]}
 - 杠杆倍数 : {v["杠杆倍数"]}
"""

        else:
            msg += "#### 当前空仓\n"

        msg += f"#### 轮动数量 : {TOP + len(SYMBOLS_WHITE) - len(SYMBOLS_BLACK)}\n"
        msg += f"#### 开仓因子 : {OPEN_LEVEL}*{OPEN_PERIOD}\n"
        # msg += f"#### 过滤因子 : {OPEN_FACTOR}>{FF_1_BASE * 100}%\n"
        msg += f"#### 平仓因子 : {CLOSE_LEVEL}*{CLOSE_PERIOD}\n"
        msg += f"#### 固定止损 : {SL_PERCENT if ENABLE_SL else 'False'}\n"
        msg += f"#### 跟踪止盈 : {TP_PERCENT if ENABLE_TP else 'False'}\n"
        msg += f"#### 资金限额 : {MAX_BALANCE * 100}%\n"

        sendMixin(msg, _type="PLAIN_POST")


def secondsToNext(exchange, level):
    levelSeconds = exchange.parseTimeframe(level.lower())
    now = int(time.time())
    seconds = levelSeconds - (now % levelSeconds)

    return seconds


def nextStartTime(level, ahead_seconds=3, offsetSec=0):
    # ahead_seconds为预留秒数, 
    # 当离开始时间太近, 本轮可能来不及下单, 因此当离开始时间的秒数小于预留秒数时, 
    # 就直接顺延至下一轮开始
    if level.endswith('m') or level.endswith('h'):
        pass
    elif level.endswith('T'):
        level = level.replace('T', 'm')
    elif level.endswith('H'):
        level = level.replace('H', 'h')
    else:
        sendAndRaise(f"{STRATEGY_NAME}: level格式错误。程序退出。")

    ti = pd.to_timedelta(level)
    now_time = dt.datetime.now()
    # now_time = dt.datetime(2019, 5, 9, 23, 50, 30)  # 修改now_time, 可用于测试
    this_midnight = now_time.replace(hour=0, minute=0, second=0, microsecond=0)
    min_step = dt.timedelta(minutes=1)

    target_time = now_time.replace(second=0, microsecond=0)

    while True:
        target_time = target_time + min_step
        delta = target_time - this_midnight
        if (
                delta.seconds % ti.seconds == 0
                and (target_time - now_time).seconds >= ahead_seconds
        ):
            # 当符合运行周期, 并且目标时间有足够大的余地, 默认为60s
            break

    target_time -= dt.timedelta(seconds=offsetSec)
    return target_time


def sleepToClose(level, aheadSeconds, test=False, offsetSec=0):
    nextTime = nextStartTime(level, ahead_seconds=aheadSeconds, offsetSec=offsetSec)
    logger.info(f"等待开始时间: {nextTime}")
    if test is False:
        time.sleep(max(0, (nextTime - dt.datetime.now()).seconds))
        while True:  # 在靠近目标时间时
            if dt.datetime.now() > nextTime:
                break
    logger.info(f"时间到,新一轮计算开始！")
    return nextTime


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getMarkets(markets):
    try:
        mks = pd.DataFrame.from_dict(markets, orient="index")
        return mks
    except Exception as e:
        logger.exception(e)
        sendAndRaise(f"{STRATEGY_NAME}: getMarkets()错误, 程序退出。{e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getTickers(exchange):
    try:
        tk = exchange.fetchTickers()
        tk = pd.DataFrame.from_dict(tk, orient="index")
        return tk
    except Exception as e:
        logger.exception(e)
        sendAndRaise(f"{STRATEGY_NAME}: getTickers()错误, 程序退出。{e}")


def getPrices(exchange):
    # 获取所有币种的ticker数据
    tickers = retryIt(exchange.fapiPublic_get_ticker_price)
    tickers = pd.DataFrame(tickers)
    tickers.set_index('symbol', inplace=True)
    tickers = tickers.astype({"price": float}, copy=True)
    return tickers['price']


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getTicker(exchange, markets, symbol):
    try:
        symbolId = markets[symbol]["id"]
        tk = exchange.fapiPublicGetTickerBookticker({"symbol": symbolId})
        tk = pd.DataFrame(tk, index=[0])
        tk = tk.astype(
            {
                "bidPrice": float,
                "bidQty": float,
                "askPrice": float,
                "askQty": float,
            }
        )
        # symbol  bidPrice bidQty  askPrice  askQty           time
        # 0  BTCUSDT  20896.70  6.719  20896.80  12.708  1673800340925
        return tk
    except Exception as e:
        logger.exception(e)
        sendAndRaise(f"{STRATEGY_NAME}: getTicker()错误, 程序退出。{e}")


def getTopN(tickers, rule="/USDT", _type="quoteVolume", n=50):
    tickers["timestamp"] = pd.to_datetime(tickers["timestamp"], unit="ms")
    tickers = tickers.filter(like=rule, axis=0)
    r = (
        tickers.set_index("timestamp")
        .sort_index()
        .last("24h")
        .nlargest(n, _type)["symbol"]
    )
    return list(r)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getBalances(exchange):
    # positions:
    # initialMargin maintMargin unrealizedProfit positionInitialMargin openOrderInitialMargin leverage  isolated entryPrice maxNotional positionSide positionAmt notional isolatedWallet updateTime bidNotional askNotional
    try:
        b = exchange.fetchBalance()["info"]
        balances = pd.DataFrame(b["assets"])
        balances.set_index("asset", inplace=True)
        balances.index.name = None
        positions = pd.DataFrame(b["positions"])
        positions.set_index("symbol", inplace=True)
        positions.index.name = None
        b.pop("assets")
        b.pop("positions")
        total = pd.DataFrame(b, index=[0])
        return total, balances, positions
    except Exception as e:
        logger.exception(e)
        sendAndRaise(f"{STRATEGY_NAME}: getBalances()错误, 程序退出。{e}")


def getBalance(exchange, asset="usdt"):
    asset = asset.upper()
    r = exchange.fapiPrivateGetAccount()["assets"]
    r = pd.DataFrame(r)
    bal = float(r.loc[r["asset"] == asset, "walletBalance"])
    return bal


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getKlines(exchangeId, level, amount, symbols):
    # getKlines要在子进程里使用, 进程之间不能直接传递ccxt实例, 因此只能在进程内部创建实例
    exchange = getattr(ccxt, exchangeId)(EXCHANGE_CONFIG)
    amount += NEW_KLINE_NUM
    klines = {}

    for symbol in symbols:
        k = exchange.fetchOHLCV(symbol, level, limit=amount)
        k = pd.DataFrame(
            k, columns=["candle_begin_time", "open", "high", "low", "close", "volume"]
        )
        k.drop_duplicates(subset=["candle_begin_time"], keep="last", inplace=True)
        k.sort_values(by="candle_begin_time", inplace=True)
        k["candle_begin_time"] = pd.to_datetime(
            k["candle_begin_time"], unit="ms"
        ) + dt.timedelta(hours=8)
        k = k[:-1]
        if len(k) + 1 < amount:
            logger.debug(f"{symbol}k线数量{len(k) + 1}少于要求{amount}, 剔除该币种")
            continue
        # k = k[["candle_begin_time", "close"]]
        klines[symbol] = k
        logger.debug(f"获取到{symbol} k线{len(k)}根")

        if len(symbols) > 1:
            time.sleep(SLEEP_SHORT)

    return klines


def getKlineForSymbol(exchange, level, amount, symbol):
    amount += NEW_KLINE_NUM

    k = exchange.fetchOHLCV(symbol, level, limit=amount)
    k = pd.DataFrame(
        k, columns=["candle_begin_time", "open", "high", "low", "close", "volume"]
    )
    k.drop_duplicates(subset=["candle_begin_time"], keep="last", inplace=True)
    k.sort_values(by="candle_begin_time", inplace=True)
    k["candle_begin_time"] = pd.to_datetime(
        k["candle_begin_time"], unit="ms"
    ) + dt.timedelta(hours=8)
    k = k[:-1]

    k = k[["candle_begin_time", "close"]]
    logger.debug(f"获取到{symbol} k线{len(k)}根")

    return k


def getKlinesMulProc(exchangeId, symbols, level, amount):
    singleGetKlines = partial(getKlines, exchangeId, level, amount)
    pNum = min(cpu_count(), len(symbols))
    logger.debug(f"开启{pNum}线程获取k线")
    with Pool(processes=pNum) as pool:
        kNew = pool.map(singleGetKlines, [[symbol] for symbol in symbols])
        kNew = [i for i in kNew if i]  # 去空
    klinesDict = {list(i.keys())[0]: list(i.values())[0] for i in kNew}

    return klinesDict


def resampleKlines(df, level):
    level = level.upper()
    if not (level.endswith("M")
            or level.endswith("H")
            or level.endswith("D")
            or level.endswith("T")):
        e = "周期格式不正确，退出。应当以M(T)、H、D结尾。"
        logger.error(e)
        raise RuntimeError(e)

    level = level.replace("M", "T")
    dfNew = df.resample(
        rule=level,
        on="candle_begin_time",
        label="left",
        closed="left"
        ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # 整理数据，很重要！可以避免后面出现的各种花式错误
    # 剔除掉原始数据中可能出现的空白数据
    dfNew.dropna(subset=["open"], inplace=True)
    dfNew = dfNew[dfNew["volume"] > 0]
    dfNew.reset_index(inplace=True)
    dfNew.drop_duplicates(subset="candle_begin_time", ignore_index=True, inplace=True)
    dfNew.sort_values(by="candle_begin_time", inplace=True)
    dfNew.reset_index(drop=True, inplace=True)

    return dfNew


def combineK(kHistory, kNew):
    if kHistory.keys() != kNew.keys():
        sendAndRaise(f"{STRATEGY_NAME}: combineK()报错：历史k线与最新k线的symbols不一致, 请检查。退出。")

    kAll = dict.fromkeys(kHistory.keys())
    for symbol in kHistory.keys():
        kAll[symbol] = pd.concat([kHistory[symbol], kNew[symbol]], ignore_index=True)
        kAll[symbol].drop_duplicates(
            subset="candle_begin_time", keep="last", inplace=True
        )
        kAll[symbol].sort_values(by="candle_begin_time", inplace=True)
        kAll[symbol].reset_index(drop=True, inplace=True)

    return kAll


def getOffset(exchange, df, holdHour, openLevel, offsetList, runtime):
    # holdTime: "4h"
    # level: "1h"
    holdHourSec = exchange.parseTimeframe(holdHour)
    openLevelSec = exchange.parseTimeframe(openLevel)
    df = df.copy()
    df["offset"] = df["candle_begin_time"].astype("int64") // 10 ** 9 % holdHourSec // openLevelSec
    df = df.loc[df["offset"].isin(offsetList)]
    df.sort_values(by="candle_begin_time", inplace=True)
    # 如果是测试, 执行时间在等待时间点之前, 为了校正offset, hh要+1, 
    # 如果是真实执行, 执行时间在时间点之后, hh不用+1
    holdHourSec += openLevelSec if IS_TEST else 0
    df = df[df["candle_begin_time"] >= (runtime - dt.timedelta(seconds=holdHourSec))]

    return df


def setFilter(kDict, _filters):
    kNew = {}
    for symbol, df in kDict.items():
        result = True
        for fName, fParas in _filters.items():
            _cls = __import__("filters")
            r = getattr(_cls, fName)(df, fParas)
            if r == False:
                result = False
                logger.debug(f"{symbol}不符合 {fName} 被剔除")
                break
        if result is True:
            kNew[symbol] = kDict[symbol]

    return kNew


def setFactor(klinesDict: dict, openFactor, openPeriod):
    kDf = pd.DataFrame()
    for symbol, df in klinesDict.items():
        df["symbol"] = symbol
        # 计算开仓factor
        nameOF = f"openFactor"
        df[nameOF] = getattr(signals, openFactor)(df, openPeriod)
        kDf = pd.concat([kDf, df], ignore_index=True)
        time.sleep(0.01)

    return kDf


def getChosen(klinesDf: pd.DataFrame, selectNum, filters=None):
    # 选币因子排名
    if not klinesDf.empty:
        klinesDf["rank"] = klinesDf.groupby("candle_begin_time")["openFactor"].rank(ascending=False)
        klinesDf.sort_values(by=["candle_begin_time", "rank"], inplace=True)
        logger.debug(f"选币排名过程:\n{klinesDf}")

        g = klinesDf.groupby("candle_begin_time")
        chosenLong = g.head(selectNum).copy()
        chosenLong["side"] = 1
        chosenLong = chosenLong[["candle_begin_time", "symbol", "side", "close"]]
        return chosenLong
        # chosenShort = g.tail(selectNum)
        # chosenShort["side"] = -1
        # chosen = pd.concat([chosenLong, chosenShort], ignore_index=True)
        # chosen = chosen[["candle_begin_time", "symbol", "side", "close"]]
        # return chosen
    else:
        return None


def getPosAim(chosen: pd.DataFrame, balance, leverage, offsetList, selectNum):
    eachCost = balance * leverage / len(offsetList) / selectNum
    logger.debug(f"每仓位使用资金: {eachCost}")
    # 计算每offset应当持有的币种数量
    chosen["amount"] = eachCost / chosen["close"] * chosen["side"]
    logger.debug(f"每个offset应当持仓数量:\n{chosen}")
    # 合并所有offset应当持有的币种数量, 计算出目标仓位
    df = chosen.groupby("symbol")[["amount"]].sum()
    df.index.name = None
    return df


def getSignal(posAim: pd.DataFrame, posNow: pd.DataFrame):
    # 获取当前持仓
    posNow = posNow[["contracts", "timestamp"]].copy()
    posNow.rename(columns={"contracts": "amount"}, inplace=True)
    logger.debug(f"当前持仓:\n{posNow}")
    # 合并当前持仓和目标持仓
    pos = pd.merge(posNow, posAim, how="outer", suffixes=("Now", "Aim"), left_index=True, right_index=True)
    pos.fillna(value=0, inplace=True)
    logger.debug(f"当前持仓与目标持仓合并后:\n{pos}")
    # 合并持仓生成交易信号
    sig = pd.DataFrame(pos["amountAim"] - pos["amountNow"], columns=["amount"]).sort_values(by="amount")
    return sig


def checkStoploss(exchange, markets, posNow: pd.DataFrame, closeFactor, closeLevel, closePeriod, closeMethod):
    pos = posNow.copy()
    symbols = pos.index.tolist()
    kDict = getKlinesMulProc(
        exchangeId=exchange.id,
        symbols=symbols,
        level=closeLevel,
        amount=closePeriod,
    )
    # 检查每个币种是否满足closeFactor的止损条件
    for symbol, df in kDict.items():
        df["symbol"] = symbol
        df["closeFactor"] = getattr(signals, closeFactor)(df, closePeriod)

        if closeMethod == "less":
            if df.iloc[-1]["close"] < df.iloc[-1]["closeFactor"]:
                closePositionForce(exchange, markets, posNow, symbol)
                sendAndPrintInfo(f"{STRATEGY_NAME} {symbol}满足closeFactor已平仓")
        elif closeMethod == "XXX":
            pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getPositions(exchange):
    # positions:
    # info    id  contracts  contractSize  unrealizedPnl  leverage liquidationPrice  collateral  notional markPrice  entryPrice timestamp  initialMargin  initialMarginPercentage  maintenanceMargin  maintenanceMarginPercentage marginRatio datetime marginMode marginType  side  hedged percentage
    try:
        p = exchange.fetchPositions()
        p = pd.DataFrame(p)
        p.set_index("symbol", inplace=True)
        p.index.name = None
        return p
    except Exception as e:
        logger.exception(e)
        sendAndRaise(f"{STRATEGY_NAME}: getPositions()错误, 程序退出。{e}")


def getOpenPosition(exchange):
    pos = getPositions(exchange)
    op = pos.loc[pos["contracts"] != 0]
    op = op.astype(
        {
            "contracts": float,
            "unrealizedPnl": float,
            "leverage": float,
            "liquidationPrice": float,
            "collateral": float,
            "notional": float,
            "markPrice": float,
            "entryPrice": float,
            "marginType": str,
            "side": str,
            "percentage": float,
            "timestamp": float,
        }
    )
    op = op[["contracts", "notional", "percentage", "unrealizedPnl", "entryPrice", "markPrice",
             "liquidationPrice", "side", "leverage", "timestamp", "marginType"]]
    return op


def getSignal_backup(klines, _openPosition, factor, para):
    # 每个币种计算因子列
    for symbol, df in klines.items():
        # 计算因子列
        df = getattr(signals, factor)(df, para)

        df.rename(
            columns={
                "open": f"{symbol}_open",
                "high": f"{symbol}_high",
                "low": f"{symbol}_low",
                "close": f"{symbol}_close",
                "volume": f"{symbol}_volume",
                "factor": f"{symbol}_factor",
            },
            inplace=True,
        )

    # 汇总每个币种的df, 生成总的dfAll
    dfs = list(klines.values())
    dfAll = reduce(lambda df1, df2: pd.merge(df1, df2, on="candle_begin_time"), dfs)

    dfAll.drop_duplicates(subset="candle_begin_time", ignore_index=True, inplace=True)
    dfAll.sort_values(by="candle_begin_time", inplace=True)
    dfAll.reset_index(drop=True, inplace=True)

    # 根据factor选币
    # 如果最大涨幅都小于0, 那么空仓
    if dfAll.iloc[-1].filter(like="factor").max() < MIN_CHANGE:
        return 0

    # .idxmax(axis=1)选取一行中的最大值的列名, 即选取最大factor的币种
    # 列名如ETH_factor, 用replace把_factor去掉
    dfAll["chosen"] = (
        dfAll.filter(like="factor").idxmax(axis=1).str.replace("_factor", "")
    )
    logger.debug(f"dfAllWithChosen:\n{dfAll.iloc[-1].filter(regex='.*factor|chosen')}")

    # 根据现有持仓, 生成交易信号
    has = _openPosition.index.tolist()[0] if len(_openPosition) else None
    logger.debug(f"has: {has}")
    new = dfAll.iloc[-1]["chosen"]
    logger.debug(f"new: {new}")

    if has != new:
        sig = {
            0: has,
            1: new,
        }
    else:
        sig = None

    return sig


def getOpenSignal(
    exchangeId,
    klines,
    selectNum,
    selectFactor,
    selectPeriod,
    filterFactor,
    filterLevel,
    filterPeriod,
):
    # 每个币种计算因子列
    for symbol, df in klines.items():
        if (df is None) or df.empty or len(df) < selectPeriod: continue
        logger.debug(symbol)
        # 计算因子列
        df = getattr(signals, selectFactor)(df, selectPeriod)
        df["symbol"] = symbol

    # 汇总每个币种的df, 生成总的dfAll
    dfs = list(klines.values())
    dfAll = reduce(lambda df1, df2: pd.concat([df1, df2], ignore_index=True), dfs)
    # 根据时间和因子排序, 最新k线的因子排序出现在最后
    g = dfAll.groupby("candle_begin_time")
    # 有些币缺少k线, 会导致最后一组k线的数量变少, 因此用最后一组k线的数量作为选币池的总个数, 过滤掉最后一组中没有出现的币种
    coins_num = g.size()[-1]
    logger.debug(f"币池总数{len(klines)}, 最新k线币总数{coins_num}")
    dfAll["rank"] = g[selectFactor].rank(ascending=False, method="first")
    dfAll.sort_values(by=["candle_begin_time", "rank"], inplace=True)
    # 最新k线的排序结果
    dfNew = dfAll.tail(coins_num)
    logger.info(
        f'本周期因子排序结果:\n{dfNew[["candle_begin_time", "symbol", selectFactor, "rank"]]}'
    )

    # 根据因子排序选前几名的币, 也可以选后几名的币
    longCoins = dfNew.head(min(selectNum, int(len(dfNew) / 2)))
    # shortCoins = dfNew.tail(min(SELECTION_NUM, int(len(dfNew)/2)))
    # 还要满足下限参数的要求
    longCoins = longCoins.loc[dfNew[selectFactor] > MIN_CHANGE]

    # 最后得出一个symbol list
    longCoins = longCoins["symbol"].values.tolist()
    filterCoins = []
    if longCoins:
        logger.debug(f"本周期符合selectFactor的币种{longCoins}")

        # 用过滤因子过滤结果
        # 满足4h close>sma20
        singleGetKlines = partial(getKlines, exchangeId, filterLevel, filterPeriod)
        pNum = min(cpu_count(), len(longCoins))
        with Pool(processes=pNum) as pool:
            klines = pool.map(singleGetKlines, [[symbol] for symbol in longCoins])
        klines = {list(i.keys())[0]: list(i.values())[0] for i in klines}

        filterCoins = []
        for symbol, df in klines.items():
            isOk = getFilterSignal(df, factor=filterFactor, period=filterPeriod)
            if isOk:
                filterCoins.append(symbol)

        if filterCoins:
            logger.debug(f"本周期符合filterFactor的币种{filterCoins}")
        else:
            logger.debug(f"本周期无符合filterFactor的币种")

    else:
        logger.debug(f"本周期无符合selectFactor的币种")

    return filterCoins


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def setMarginType(exchange, symbolId, _type=1):
    if _type == 1:
        t = "CROSSED"
    elif _type == 2:
        t = "ISOLATED"

    p = {
        "symbol": symbolId,
        "marginType": t,
    }

    try:
        exchange.fapiPrivatePostMargintype(p)
    except ccxt.MarginModeAlreadySet:
        pass


def getOrderPrice(symbol, price, action, markets):
    precision = markets.loc[symbol, "precision"]["price"]

    if action == 1:
        orderPrice = price * (1 + SLIPPAGE)
    elif action == 0:
        orderPrice = price * (1 - SLIPPAGE)

    orderPrice = int(orderPrice * (10 ** precision)) / (10 ** precision)
    # orderPrice = exchange.priceToPrecision(symbol, orderPrice)
    logger.debug(
        f"symbol:{symbol}, slippage:{SLIPPAGE}, price:{price}, pre:{precision}, oP:{orderPrice}"
    )

    return orderPrice


def getOrderSize(symbol, action, price, balance, markets, positions):
    # 如果是卖单则直接返回现有持仓数量
    if action == 0:
        return abs(float(positions.loc[markets.loc[symbol, "id"], "positionAmt"]))

    # 如果是买单, 则根据余额计算数量
    precision = markets.loc[symbol, "precision"]["amount"]
    minLimit = markets.loc[symbol, "limits"]["amount"]["min"]
    maxLimit = markets.loc[symbol, "limits"]["market"]["max"]

    size = max(balance * LEVERAGE / price, minLimit)
    size = min(size, maxLimit)
    size = int(size * (10 ** precision)) / (10 ** precision)
    logger.debug(
        f"symbol:{symbol}, maxBalance:{MAX_BALANCE}, price:{price}, pre:{precision}, size:{size}, min:{round(0.1 ** precision, precision)}, minLimit:{minLimit}, maxLimit:{maxLimit}"
    )
    if precision == 0:
        size = int(size)
        return max(size, minLimit)
    else:
        return max(size, round(0.1 ** precision, precision))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(SLEEP_SHORT),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
def getOrderStatus(exchange, symbolId, orderId):
    return exchange.fapiPrivateGetOrder(
        {
            "symbol": symbolId,
            "orderId": orderId,
        }
    )["status"]


def placeOrder(exchange, markets, prices, signal, leverage, marginType):
    orderParams = []
    orderResp = []

    # 构造订单参数
    for symbol, row in signal.iterrows():
        try:
            symbolId = markets[symbol]["id"]
            amount = row["amount"]
            price = prices[symbolId]
            if amount > 0 and price * amount < 7:
                logger.debug(f"{symbol}买入金额小于7U,跳过该币种")
                continue

            price = price * (1 + SLIPPAGE) if amount > 0 else price * (1 - SLIPPAGE)
            price = exchange.priceToPrecision(symbol, price)
            side = "BUY" if amount > 0 else "SELL"
            reduceOnly = "true" if side == "SELL" else "false"
            try:
                amount = exchange.amountToPrecision(symbol, abs(amount))
            except ccxt.ArgumentsRequired as e:
                logger.debug(f"{symbol}下单量不满足最小限制,跳过该币种{e}")
                continue
            para = {
                "symbol": symbolId,
                "side": side,
                "type": "LIMIT",
                "price": price,
                "quantity": amount,
                "reduceOnly": reduceOnly,
                "timeInForce": "GTC",
            }
            logger.debug(f"{symbol}订单参数:{para}")
            orderParams.append(para)
            # 设置杠杆和全仓模式
            retryIt(exchange.fapiPrivatePostLeverage, paras={"symbol": symbolId, "leverage": leverage})
            retryIt(exchange.fapiPrivatePostMargintype, paras={"symbol": symbolId, "marginType": marginType})
        except Exception as e:
            sendAndPrintError(f"{STRATEGY_NAME} {symbol}构造订单信息出错,跳过该币种:{e}")
            logger.exception(e)
            continue

    # 发送订单
    _n = "\n"
    for i in range(0, len(orderParams), 5):
        _orderP = orderParams[i: i + 5]
        logger.debug(f"本次批量下单参数:\n{_n.join(map(str, _orderP))}")
        r = retryIt(
            exchange.fapiPrivatePostBatchorders,
            paras={"batchOrders": json.dumps(_orderP)},
            critical=True
        )
        logger.debug(f"本次下单返回结果:\n{_n.join(map(str, r))}")

        # 检查订单状态
        for idx, v in enumerate(r):
            if "orderId" in v:
                orderResp.append(v)
            else:
                sendAndPrintError(f"{STRATEGY_NAME} 下单出错: {_orderP[idx]}  {v}")

    return orderResp


def getFilterSignal(df, factor, period):
    factor = factor.lower()
    factorName = f"sma{period}"

    if factor == "closegtsma":
        df[factorName] = df["close"].rolling(period).mean()
        c1 = df.iloc[-1]["close"]
        f1 = df.iloc[-1][factorName]

        return True if c1 > f1 else False

    elif factor == "xxx":
        pass


def getCloseSignal(df, factor, period, method):
    factor = factor.lower()
    period = int(period)
    method = method.lower()

    if factor == "sma":
        df["factor"] = df["close"].rolling(period).mean()
    elif factor == "xxx":
        pass

    c1 = df.iloc[-1]["close"]
    f1 = df.iloc[-1]["factor"]

    if method == "less":
        r = c1 < f1
    elif method == "xxx":
        pass

    return r


def placeBatchOrderClose(exchange, symbols, markets):
    # symbols = [{symbol:posAmount},{symbol:posAmount},...]

    # batchOrder每批最多5笔订单, 分批发送
    ordersTotal = []
    responsesTotal = []
    closed = []
    for i in range(0, len(symbols), 5):
        subSymbols = symbols[i: i + 5]

        orders = []
        for s in subSymbols:
            try:
                symbol = list(s.keys())[0]
                amount = float(list(s.values())[0])
                symbolId = markets[symbol]["id"]
                orders.append(
                    {
                        "symbol": symbolId,
                        "side": "SELL",
                        "type": "MARKET",
                        "quantity": exchange.amountToPrecision(symbol, amount),
                        "reduceOnly": "true",
                    }
                )
            except Exception as e:
                sendAndPrintError(
                    f"{STRATEGY_NAME} placeBatchOrderClose({s}构造订单报错, 跳过该币种, 请检查日志{e})"
                )
                logger.exception(e)
                continue

        logger.debug(f"批量平仓单参数: {orders}")
        ordersTotal += orders
        try:
            ordersStr = [
                exchange.encode_uri_component(exchange.json(order), safe=",")
                for order in orders
            ]
            response = exchange.fapiPrivatePostBatchOrders(
                {'batchOrders': '[' + ','.join(ordersStr) + ']'}
            )

            responsesTotal += response

        except Exception as e:
            sendAndCritical(e)
            logger.exception(e)

    for index, r in enumerate(responsesTotal):
        for i in range(MAX_TRY):
            time.sleep(SLEEP_SHORT)

            if "orderId" in r:
                orderInfo = exchange.fapiPrivateGetOrder(
                    {"orderId": r["orderId"], "symbol": r["symbol"]}
                )
                if orderInfo["status"] == "FILLED":
                    closed.append(r["symbol"])
                    logger.debug(f"{r['symbol']}平仓成功")
                    break
                else:
                    if i == MAX_TRY - 1:
                        sendAndCritical(f"{r['symbol']}平仓查询三次仍不成功, 请检查")
                        break
            else:
                sendAndCritical(
                    f"{STRATEGY_NAME} placeBatchOrderClose({orders[index]['symbol']})在批量平仓中失败, 请检查。{r}"
                )

    return closed


def placeBatchOrderOpen(exchange, symbols, markets, selectNum):
    # symbols = [a,b,c]
    balance = getBalance(exchange, symbol="USDT")["free"] * MAX_BALANCE
    # 本次使用的余额=总余额/未建仓数量
    eachCash = int(balance / selectNum)

    # batchOrder每批最多5笔订单, 分批发送
    ordersTotal = []
    responsesTotal = []
    opened = []
    for i in range(0, len(symbols), 5):
        subSymbols = symbols[i: i + 5]

        orders = []
        for symbol in subSymbols:
            symbolId = markets[symbol]["id"]
            try:
                setMarginType(exchange, symbolId, _type=1)
                exchange.setLeverage(LEVERAGE, symbol)
            except Exception:
                pass

            try:
                if eachCash <= 5:
                    sendAndPrintInfo(f"{symbol}下单金额小于5U, 跳过")
                    continue
                orders.append(
                    {
                        "symbol": symbolId,
                        "side": "BUY",
                        "type": "MARKET",
                        "quantity": exchange.amountToPrecision(
                            symbol,
                            eachCash / getTicker(exchange, markets, symbol).loc[0, "askPrice"],
                        ),
                    }
                )
            except Exception as e:
                sendAndPrintError(
                    f"{STRATEGY_NAME} placeBatchOrderOpen({symbol}构造订单报错, 跳过该币种, 请检查日志{e})"
                )
                logger.exception(e)
                continue

        if not orders: continue
        logger.debug(f"批量开仓单参数: {orders}")
        ordersTotal += orders
        try:
            ordersStr = [
                exchange.encode_uri_component(exchange.json(order), safe=",")
                for order in orders
            ]
            response = exchange.fapiPrivatePostBatchOrders(
                {'batchOrders': '[' + ','.join(ordersStr) + ']'}
            )

            responsesTotal += response

        except Exception as e:
            sendAndCritical(e)
            logger.exception(e)

    for index, r in enumerate(responsesTotal):
        for i in range(MAX_TRY):
            time.sleep(SLEEP_SHORT)

            if "orderId" in r:
                orderInfo = exchange.fapiPrivateGetOrder(
                    {"orderId": r["orderId"], "symbol": r["symbol"]}
                )
                if orderInfo["status"] == "FILLED":
                    opened.append(r["symbol"])
                    logger.debug(f"{r['symbol']}买入成功")
                    break
                else:
                    if i == MAX_TRY - 1:
                        sendAndCritical(f"{r['symbol']}买入单查询三次仍不成功, 请检查")
                        break
            else:
                sendAndCritical(
                    f"{STRATEGY_NAME} placeBatchOrderClose({orders[index]['symbol']})在批量买入中失败, 请检查。{r}"
                )

    return opened


def openPosition(
    exchangeId,
    markets,
    openPositions,
    openFactor,
    openLevel,
    openPeriod,
    filterFactor,
    filterLevel,
    filterPeriod,
):
    ex = getattr(ccxt, exchangeId)(EXCHANGE_CONFIG)
    ex.loadMarkets()
    openedSymbols = openPositions.index.tolist()

    tickers = getTickers(ex)
    symbols = getTopN(tickers, rule=RULE, _type=TYPE, n=TOP)
    symbols = (set(symbols) | set(SYMBOLS_WHITE)) - set(SYMBOLS_BLACK)
    symbols = list(symbols)

    singleGetKlines = partial(getKlines, exchangeId, openLevel, openPeriod)
    pNum = min(cpu_count(), len(symbols))
    with Pool(processes=pNum) as pool:
        kNew = pool.map(singleGetKlines, [[symbol] for symbol in symbols])
    kNew = {list(i.keys())[0]: list(i.values())[0] for i in kNew}

    longCoins = getOpenSignal(
        exchangeId=exchangeId,
        klines=kNew,
        selectNum=SELECTION_NUM,
        selectFactor=openFactor,
        selectPeriod=openPeriod,
        filterFactor=filterFactor,
        filterLevel=filterLevel,
        filterPeriod=filterPeriod,
    )

    # 去掉选出的币与持仓币重复的, 只补足与要求选币数量的差异
    longCoins = list(set(longCoins) - set(openedSymbols))
    selectNumThisTime = SELECTION_NUM - len(openedSymbols)
    longCoins = longCoins[:selectNumThisTime]
    opened = []

    if longCoins:
        opened = placeBatchOrderOpen(ex, longCoins, markets, selectNum=selectNumThisTime)

    return opened


def closePosition(
    exchangeId, markets, openPositions, level, factor, period, method, holdTime
):
    ex = getattr(ccxt, exchangeId)(EXCHANGE_CONFIG)
    ex.loadMarkets()
    symbols = openPositions.index.tolist()
    amounts = openPositions.contracts.tolist()
    pos = {symbols[i]: amounts[i] for i in range(len(symbols))}
    klines = getKlines(exchangeId, level=level, amount=period, symbols=symbols)
    willClose = []
    closed = []

    for symbol, df in klines.items():
        closeSig = getCloseSignal(df, factor, period, method)
        reachHoldTime = (
                                time.time() * 1000 - openPositions.loc[symbol, "timestamp"]
                        ) >= ex.parseTimeframe(holdTime) * 1000
        if closeSig or reachHoldTime:
            if reachHoldTime:
                sendAndPrintInfo(f"{STRATEGY_NAME} {symbol}满足持仓时间平仓")
            if closeSig:
                sendAndPrintInfo(f"{STRATEGY_NAME} {symbol}满足closeFactor平仓")
            willClose.append({symbol: pos[symbol]})

    if willClose:
        logger.debug(f"willClose: {willClose}")
        closed = placeBatchOrderClose(ex, willClose, markets)

    return closed


def closePositionForce(exchange, markets, openPositions, symbol=None):
    # 如果没有symbol参数, 清空所有持仓, 如果有symbol只平仓指定币种
    for s, pos in openPositions.iterrows():
        if symbol is not None and s != symbol: continue
        symbolId = markets[s]["id"]
        para = {
            "symbol": symbolId,
            "side": "SELL",
            "type": "MARKET",
            "quantity": pos["contracts"],
            "reduceOnly": True,
        }

        retryIt(exchange.fapiPrivatePostOrder, paras=para, critical=True)


if __name__ == "__main__":
    exId = EXCHANGE
    ex = ccxt.binance(EXCHANGE_CONFIG)
    mkts = ex.loadMarkets()
    b = getTicker(ex, mkts, "BTC/USDT")
    print(b)
