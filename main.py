# this is code for offset branch
from multiprocessing import current_process
from multiprocessing import Pool as PPool

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
    rptpool = PPool(1)
    rptpool.apply_async(reporter, args=(EXCHANGE, REPORT_INTERVAL))
    _n = "\n"

    # 开始运行策略
    while True:

        # 获取币池列表
        mkts = ex.loadMarkets()
        logger.info(f"获取币种信息完毕, 共 {len(mkts)}")
        # 获取当前余额
        balance = getBalance(exchange=ex, asset="usdt")
        logger.info(f"获取余额信息完毕, 当前资金 {round(balance,2)}, "
                    f"可开资金 {round(balance*MAX_BALANCE,2)}, "
                    f"单币杠杆率 {LEVERAGE}, 实际杠杆率 {round(MAX_BALANCE*LEVERAGE,2)}")
        balance *= MAX_BALANCE

        # 获取当前持仓
        posNow = getOpenPosition(exchange=ex)
        logger.info(f"获取当前持仓信息完毕, 当前持仓\n{posNow}")

        # 等待下一轮开始, 获取下一轮运行时间
        runtime = sleepToClose(
            level=OPEN_LEVEL,
            aheadSeconds=AHEAD_SEC,
            offsetSec=OFFSET_SEC,
            test=IS_TEST,
        )

        # 检查止损
        if not posNow.empty and SKIP_TRADE is False:
            checkStoploss(
                exchange=ex,
                markets=mkts,
                posNow=posNow,
                closeFactor=CLOSE_FACTOR,
                closeLevel=CLOSE_LEVEL,
                closePeriod=CLOSE_PERIOD,
                closeMethod=CLOSE_METHOD,
            )

        # 从币池列表中按照TYPE排序, 取topN
        tickers = getTickers(exchange=ex)
        symbols = getTopN(
            tickers=tickers,
            rule=RULE,
            _type=TYPE,
            n=TOP,
        )
        symbols = sorted(list((set(symbols) | set(SYMBOLS_WHITE)) - set(SYMBOLS_BLACK)))
        logger.info(f"Top{TOP}筛选和合并黑白名单之后, 币池列表共 {len(symbols)}")
        # logger.debug(f"币池列表:\n{_n.join(symbols)}")

        # 多进程获取topN币种的k线数据
        kDict = getKlinesMulProc(
            exchangeId=exId,
            symbols=symbols,
            level=OPEN_LEVEL,
            amount=len(OFFSET_LIST) + OPEN_PERIOD,
        )
        logger.info(f"共获取合格的k线币种 {len(kDict)}")

        # 过滤因子筛选币池
        kDict = setFilter(kDict, _filters=FILTER_FACTORS)
        logger.info(f"过滤因子筛选后的币种 {len(kDict)}")
        if len(kDict) == 0:
            sendAndPrintInfo(f"{STRATEGY_NAME} 本周期无合格币种，等待下一周期\n\n\n")
            continue

        # 计算开仓和平仓因子
        kDf = setFactor(
            klinesDict=kDict,
            openFactor=OPEN_FACTOR,
            openPeriod=OPEN_PERIOD,
        )
        # logger.debug(f"计算开仓平仓因子结果:\n{kDf}")

        # 计算选币结果
        kDf = getChosen(
            klinesDf=kDf,
            selectNum=SELECTION_NUM,
            filters=FILTER_FACTORS,
        )
        logger.debug(f"选币结果:\n{kDf}")

        # 计算offset
        kDf = getOffset(
            exchange=ex,
            df=kDf,
            openLevel=OPEN_LEVEL,
            holdHour=HOLD_TIME,
            offsetList=OFFSET_LIST,
            runtime=runtime,
        )
        logger.info(f"计算offset结果:\n{kDf}")

        # 计算目标持仓
        logger.info(f"当前可用资金共: {balance}")
        posAim = getPosAim(
            chosen=kDf,
            balance=balance,
            leverage=LEVERAGE,
            offsetList=OFFSET_LIST,
            selectNum=SELECTION_NUM,
        )
        logger.info(f"目标持仓:\n{posAim}")

        # 目标持仓与当前持仓合并, 成为交易信号
        posNow = getOpenPosition(exchange=ex)
        sig = getSignal(posAim, posNow)
        logger.info(f"当前持仓与目标持仓合并后的交易信号:\n{sig}")

        # 根据交易信号执行订单
        if not sig.empty and SKIP_TRADE is False:
            prices = getPrices(exchange=ex)
            ordersResp = placeOrder(
                exchange=ex,
                markets=mkts,
                prices=prices,
                signal=sig,
                leverage=LEVERAGE,
                marginType=MARGIN_TYPE,
            )
            if ordersResp:
                sendAndPrintInfo(f"{STRATEGY_NAME} 本周期下单结果:\n{ordersResp}")
            else:
                logger.info(f"本周期无下单结果")

        del mkts, balance, posNow, posAim, tickers, symbols, kDict, kDf, sig
        logger.info(f"进入下一周期\n\n\n")
        if IS_TEST: exit()


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            sendAndPrintError(f"{STRATEGY_NAME} 主程序异常, 自动重启, 请检查{e}")
            logger.exception(e)
            continue

