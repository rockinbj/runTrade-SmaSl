import datetime as dt
import json
from functools import partial
from functools import reduce
# from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool as TPool

import ccxt
import pandas as pd
import requests
from tenacity import *

import signals
from exchangeConfig import *
from logger import *
from settings import *


# pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 100)  # 最多显示数据的行数
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
        r = requests.post(url, data=value, timeout=2).text
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
        msg += f"#### 选币数量 : {SELECTION_NUM}\n"
        msg += f"#### 选币分组 : {OFFSET_LIST}\n"
        msg += f"#### 持仓时间 : {HOLD_TIME}\n"
        msg += f"#### 开仓因子 : {OPEN_FACTOR}: {OPEN_LEVEL}*{OPEN_PERIOD}\n"
        msg += f"#### 过滤因子 : {list(FILTER_FACTORS.keys())[0]}: {OPEN_LEVEL} {OPEN_PERIOD}\n"
        msg += f"#### 过滤因子 : {list(FILTER_FACTORS.keys())[1]}: {CLOSE_LEVEL} {CLOSE_PERIODS[0]}\n"
        msg += f"#### 平仓因子 : {CLOSE_LEVEL} close<{CLOSE_FACTOR}{CLOSE_PERIODS[0]}\n"
        msg += f"#### 账户余额 : {round(bal, 2)}U\n"
        msg += f"#### 页面杠杆 : {LEVERAGE}\n"
        msg += f"#### 资金上限 : {MAX_BALANCE * 100}%\n"
        msg += f"#### 实际杠杆 : {round(LEVERAGE * MAX_BALANCE, 2)}\n"

        r = sendMixin(msg, _type="PLAIN_POST")


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
    # amount += NEW_KLINE_NUM
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
            logger.debug(f"{symbol} k线数量 {len(k) + 1} 少于要求 {amount}, 剔除该币种")
            continue
        # k = k[["candle_begin_time", "close"]]
        klines[symbol] = k
        logger.debug(f"获取到 {symbol} k线 {len(k)}")

        if len(symbols) > 1:
            time.sleep(SLEEP_SHORT)

    return klines


def getKlineForSymbol(exchange, level, amount, symbol):
    # amount += NEW_KLINE_NUM

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
    # pNum = min(cpu_count(), len(symbols))
    pNum = len(symbols)
    logger.info(f"开启 {pNum} 线程获取k线")
    with TPool(processes=pNum) as pool:
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


def setBeforeFilter(kDict, _filters, posNow):
    posNowList = posNow.index.tolist()

    for symbol, df in kDict.copy().items():
        if symbol in posNowList:
            logger.debug(f"{symbol} 已持仓，不进行前置过滤")
            continue

        r = True
        for fName, fParas in _filters.items():
            _cls = __import__("filters")
            _, rNow = getattr(_cls, fName)(df, fParas)
            if rNow == False:
                r = False
                logger.debug(f"{symbol} 不满足前置过滤 {fName} 被剔除")
                break

        if r == False:
            del kDict[symbol]

    return kDict


def setAfterFilter(kDict, _filters):
    kDf = pd.DataFrame()
    for symbol, df in kDict.copy().items():
        _dfList = []
        for fName, fParas in _filters.items():
            _cls = __import__("filters")
            _df, _ = getattr(_cls, fName)(df, fParas)
            # 将时间作为索引，后面需要根据时间来取交集
            _df = _df.reset_index().set_index("candle_begin_time", drop=True)
            _dfList.append(_df)
        # filter完之后取交集, 交集中有两表的重复列，
        # T旋转-->去重-->T再旋转回来，就把重复列去掉了，nb！
        kDict[symbol] = pd.concat(_dfList, axis=1, join="inner").T.drop_duplicates().T
        kDict[symbol].reset_index(inplace=True)

        if kDict[symbol].empty or kDict[symbol] is None:
            # 这里有删除键值的动作，在遍历dict的循环中不允许改变dict的键值数量
            # 因此使用了kDict.copy().items()，
            # 或者使用list(kDict)，相当于遍历dict keys的列表，就不受限制了
            # 需要用到df作为原始数据，因此选择第一种办法
            del kDict[symbol]
            continue
        else:
            kDict[symbol].sort_values("candle_begin_time", inplace=True)
            kDf = pd.concat([kDf, kDict[symbol]], ignore_index=True)

    return kDf


def setFactor(klinesDict: dict, openFactor, openPeriod):
    for symbol, df in klinesDict.items():
        df.sort_values("candle_begin_time", inplace=True)
        df["symbol"] = symbol
        # 计算开仓factor
        nameOF = f"openFactor"
        df[nameOF] = getattr(signals, openFactor)(df, openPeriod)
        klinesDict[symbol] = df

    return klinesDict


def getChosen(klinesDf: pd.DataFrame, selectNum, filters=None):
    # 选币因子排名
    if not klinesDf.empty:
        klinesDf.sort_values("candle_begin_time", inplace=True)
        klinesDf["rank"] = klinesDf.groupby("candle_begin_time")["openFactor"].rank(ascending=False)
        klinesDf.sort_values(by=["candle_begin_time", "rank"], inplace=True)
        # pd.set_option('display.max_rows', len(klinesDf) + 100)
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
    logger.info(f"每仓位使用资金: {eachCost}")
    # 计算每offset应当持有的币种数量
    chosen["amount"] = eachCost / chosen["close"] * chosen["side"]
    logger.info(f"每个offset应当持仓数量:\n{chosen}")
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


def checkStoploss(exchange, markets, posNow: pd.DataFrame,
                  closeFactor, closeLevel, closeMethod,
                  closePeriods):
    pos = posNow.copy()
    symbols = pos.index.tolist()
    kDict = getKlinesMulProc(
        exchangeId=exchange.id,
        symbols=symbols,
        level=closeLevel,
        amount=max(closePeriods)+100,  # 要根据最大均线周期定数量，如果用EMA需要大量k线
    )
    # 检查每个币种是否满足closeFactor的止损条件
    for symbol, df in kDict.items():
        df["symbol"] = symbol
        names = []
        _stop = False
        for k, v in enumerate(closePeriods):
            name = closeFactor + str(k)
            names.append(name)
            df[name] = getattr(signals, closeFactor)(df, v)

        if closeMethod == "less":
            # 要求CLOSE_METHOD=[]列表元素数量为1
            if df.iloc[-1]["close"] < df.iloc[-1][names[0]]:
                _stop = True
        elif closeMethod == "sma1LtSma2":
            # 要求CLOSE_METHOD=[]列表元素数量为2
            if df.iloc[-1][names[0]] < df.iloc[-1][names[1]]:
                _stop = True
        elif closeMethod == "xxx":
            pass

        if _stop is True:
            closePositionForce(exchange, markets, posNow, symbol)
            sendAndPrintInfo(f"{STRATEGY_NAME} {symbol} 满足平仓因子{CLOSE_FACTOR}:{CLOSE_METHOD}已平仓")
            pos = getOpenPosition(exchange)

    return pos


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
    op = pd.DataFrame()
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
    op = op[["contracts", "notional", "percentage", "unrealizedPnl", "leverage", "entryPrice", "markPrice",
             "liquidationPrice", "datetime", "side", "marginType", "timestamp"]]
    return op


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

    # 整理订单回执，去掉无用信息
    # Order Response Format:
    # [{'orderId': '20901193940', 'symbol': 'LTCUSDT', 'status': 'FILLED', 'clientOrderId': 'Rock1674768603670',
    # 'price': '85.39', 'avgPrice': '87.13000', 'origQty': '10.087', 'executedQty': '10.087',
    # 'cumQuote': '878.88031', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': True,
    # 'closePosition': False, 'side': 'SELL', 'positionSide': 'LONG', 'stopPrice': '0',
    # 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'time': '1674768603876',
    # 'updateTime': '1674768603876'}, ...]
    if orderResp:
        for k, v in enumerate(orderResp):
            orderResp[k] = [v["side"], v["symbol"], v["cumQuote"]]
        orderResp = "\n".join([str(i) for i in orderResp])
    return orderResp


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
