import ccxt
from functions import *
from exchangeConfig import *

ex = getattr(ccxt, EXCHANGE)(EXCHANGE_CONFIG)
mkts = ex.loadMarkets()

pos = getOpenPosition(ex)
logger.info(f"当前持仓情况:\n{pos}")

closePositionForce(ex, mkts, pos)

pos = getOpenPosition(ex)
logger.info(f"强制平仓后的持仓情况:\n{pos}")


