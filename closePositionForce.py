import sys
import ccxt
from functions import *
from exchangeConfig import *


print("如果需要清仓，请加入'--close'参数\n")

ex = getattr(ccxt, EXCHANGE)(EXCHANGE_CONFIG)
mkts = ex.loadMarkets()

pos = getOpenPosition(ex)
pos = pos[["contracts", "notional", "unrealizedPnl", "percentage", "leverage", "entryPrice", "markPrice",
           "liquidationPrice", "marginType", "side"]]
print(f"当前持仓情况:\n{pos}\n")
bal = getBalance(ex, asset="USDT")
print(f"当前余额:\n{bal}\n")


if "--close" in sys.argv:
    print(f"\n\n\n执行清仓操作。。。")
    closePositionForce(ex, mkts, pos)

    pos = getOpenPosition(ex)
    pos = pos[["contracts", "notional", "unrealizedPnl", "percentage", "leverage", "entryPrice", "markPrice",
               "liquidationPrice"]]
    print(f"当前持仓情况:\n{pos}")
    bal = getBalance(ex)
    print(f"当前余额:\n{bal}")
