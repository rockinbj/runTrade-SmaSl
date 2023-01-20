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

    # 开启一个非阻塞的报告进程
    rptpool = Pool(1)
    rptpool.apply_async(reporter, args=(EXCHANGE, REPORT_INTERVAL))
    
    # 开始运行策略
    while True:
        startTime = time.time()
        exchangeId = EXCHANGE
        mkts = ex.loadMarkets()

        # 先检查是否有需要平仓的
        pos = getOpenPosition(ex)
        logger.info(f"当前持仓币种: {pos.index.tolist()}")
        closed = closePosition(
            exchangeId=exchangeId,
            markets=mkts,
            openPositions=pos,
            level=CLOSE_LEVEL,
            factor=CLOSE_FACTOR,
            period=CLOSE_PERIOD,
            method=CLOSE_METHOD,
            holdTime=HOLD_TIME,
        )
        
        # 如果有平仓动作就重新检查持仓数量
        if closed:
            sendAndPrintInfo(f"{STRATEGY_NAME} 本周期平仓: {closed}")
            pos = getOpenPosition(ex)
            currentCoinNum = pos.shape[0]
        else:
            logger.info(f"本周期无平仓币种")
        
        # 检查当前持仓状态
        pos = getOpenPosition(ex)
        currentCoinNum = pos.shape[0]
        needCoinNum = reload(settings).SELECTION_NUM
        logger.info(f"当前持仓币种: {pos.index.tolist()}")
        logger.info(f"当前持币: {currentCoinNum}, 选币数量: {needCoinNum}")

        # 如果当前持仓小于选币数量，则选币做多
        if currentCoinNum < needCoinNum:
            opened = openPosition(
                exchangeId=exchangeId,
                markets=mkts,
                openPositions=pos,
                openFactor=OPEN_FACTOR,
                openLevel=OPEN_LEVEL,
                openPeriod=OPEN_PERIOD,
                filterFactor=FILTER_FACTOR,
                filterLevel=CLOSE_LEVEL,
                filterPeriod=CLOSE_PERIOD,
            )

            if opened:
                sendAndPrintInfo(f"{STRATEGY_NAME} 本周期开仓: {opened}")
            else:
                logger.info(f"本周期无开仓币种")
                
        logger.debug(f"本轮用时:{round(time.time()-startTime, 2)}")
        # 等待下一轮
        sleepToClose(level=OPEN_LEVEL, aheadSeconds=AHEAD_SEC, test=IS_TEST, offsetSec=OFFSET_SEC)


if __name__ == "__main__":
    main()
