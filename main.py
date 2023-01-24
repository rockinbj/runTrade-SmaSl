# this is code for offset branch
import time
from functools import partial
from importlib import reload
from multiprocessing import Pool, cpu_count, current_process
from os import makedirs, path

import pandas as pd

import settings
from exchangeConfig import *
from functions import *
from settings import *

pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)
logger = logging.getLogger("app.main")


def reporter(exchangeId, interval):
    process = current_process()
    process.name = "Reporter"
    while True:
        sendReport(exchangeId, interval)
        time.sleep(0.5)


def main():
    ex = getattr(ccxt, EXCHANGE)(EXCHANGE_CONFIG)
    exId = EXCHANGE

    # 开启一个非阻塞的报告进程
    rptpool = Pool(1)
    rptpool.apply_async(reporter, args=(EXCHANGE, REPORT_INTERVAL))

    # 开始运行策略
    while True:
        # 获取币池列表
        mkts = ex.loadMarkets()

        # 等待下一轮开始，获取下一轮运行时间
        runtime = sleepToClose(
                level=OPEN_LEVEL,
                aheadSeconds=AHEAD_SEC,
                test=IS_TEST,
                offsetSec=OFFSET_SEC,
        )

        # 从币池列表中按照TYPE排序，取topN
        tickers = getTickers(exchange=ex)
        symbols = getTopN(
            tickers=tickers,
            rule=RULE,
            _type=TYPE,
            n=TOP,
        )
        _n = "\n"
        logger.info(f"币池列表共 {len(symbols)}:\n{_n.join(symbols)}")

        # 获取topN币种的k线数据
        klinesDict = getKlines(
            exchangeId=exId,
            level=OPEN_LEVEL,
            amount=OPEN_PERIOD+len(OFFSET_LIST),
            symbols=symbols
        )
        logger.info(f"共获取合格的k线数据 {len(klinesDict)}")

        # 计算开仓和平仓因子
        klinesDict = setFactor(
                exchangeId=exId,
                klinesDf=klinesDict,
                openFactor=OPEN_FACTOR,
                openPeriod=OPEN_PERIOD,
                closeFactor=CLOSE_FACTOR,
                closeLevel=CLOSE_LEVEL,
                closePeriod=CLOSE_PERIOD,
        )
        logger.debug(f"计算开仓平仓因子结果:\n{klinesDict}")

        # 计算offset
        klinesDf = setOffset(
                klinesDict=klinesDict,
                holdTime=HOLD_TIME,
                offsetList=OFFSET_LIST,
                runtime=runtime,
        )
        logger.debug(f"计算offset结果:\n{klinesDict}")

        # 每个offset计算选币结果
        chosen = getChosen(
                klinesDf=klinesDf,
                filterFactor1=FILTER_FACTOR_1,
                filterFactor2=FILTER_FACTOR_2,
                selectNum=SELECTION_NUM,
        )
        logger.info(f"计算每个offset选币结果:\n{chosen}")

        # 计算目标持仓
        balance = getBalance(exchange=ex, symbol="usdt")["free"]
        logger.info(f"当前可用资金共: {balance}")
        posAim = getPosAim(
                chosen=chosen,
                balance=balance,
                leverage=LEVERAGE,
                offsetList=OFFSET_LIST,
                selectNum=SELECTION_NUM,
        )
        logger.info(f"目标持仓:\n{posAim}")

        # 获取当前持仓
        posNow = getOpenPosition(exchange=ex)
        # 目标持仓与当前持仓合并，成为交易信号
        sig = getSignal(posAim, posNow)
        logger.info(f"交易信号:\n{sig}")



if __name__ == "__main__":
    main()
