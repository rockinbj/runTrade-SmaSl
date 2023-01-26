# 策略命名
STRATEGY_NAME = "鹤-Sma-Test"
IS_TEST = False

# 轮动池黑白名单,格式"BTC/USDT"
SYMBOLS_WHITE = []
SYMBOLS_BLACK = ["BTCDOM/USDT", "DEFI/USDT", "BLUEBIRD/USDT"]

# 选币范围
# 只选取USDT交易对的成交量topN
RULE = "/USDT"
TYPE = "quoteVolume"
TOP = 200

# 选币个数
SELECTION_NUM = 1

# 开仓参数，4h*3，例如12小时bias因子涨幅排行
OPEN_FACTOR = "s_bias"
OPEN_LEVEL = "5m"
OPEN_PERIOD = 6

# 持仓时间，offset
HOLD_TIME = "30m"
OFFSET_LIST = [0,1,2,3,4,5]

# 平仓参数，例如4h级别close小于MA30
CLOSE_LEVEL = "30m"
CLOSE_FACTOR = "s_sma"
CLOSE_PERIOD = 12
CLOSE_METHOD = "less"

# 过滤因子，涨幅下限，小于此则排除
FILTER_FACTORS = {
    "f_bias": {
        "level": OPEN_LEVEL,
        "period": OPEN_PERIOD,
        "base": 0 / 100,
        "gt": True,
    },
    "f_sma": {
        "level": CLOSE_LEVEL,
        "period": CLOSE_PERIOD,
        "gt": True,
    }
}

# 涨幅下限，小于此空仓
MIN_CHANGE = 0 / 100
# 页面杠杆率
LEVERAGE = 2
# 资金上限，控制实际杠杆率
MAX_BALANCE = 50 / 100
# 保证金模式
MARGIN_TYPE = "CROSSED"
# MARGIN_TYPE = "ISOLATED"
# 限价单安全滑点
SLIPPAGE = 1.5 / 100


# 跟踪止盈开关，跟踪比例，注意TP和SL的比例都是价格比例，
# 如果价格波动1%，2倍杠杆，那么ROE的波动就是2%
ENABLE_TP = False
TP_PERCENT = 3/100
# 止损开关，开仓价的向下比例
ENABLE_SL = False
SL_PERCENT = 3/100



# 本轮开始之前的预留秒数，小于预留秒数则顺延至下轮
AHEAD_SEC = 3
# 向前偏移秒数，为了避免在同一时间下单造成踩踏
OFFSET_SEC = 0

# 获取最新k线的数量
NEW_KLINE_NUM = 180


# 输出目录
import datetime as dt
OUTPUT_PATH = "output"
TIME_PATH = dt.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


# 设置Log
import logging
LOG_PATH = "log"
LOG_LEVEL_CONSOLE = logging.DEBUG
LOG_LEVEL_FILE = logging.DEBUG


# 报告发送间隔分钟
REPORT_INTERVAL = 30
# 是否拨打电话
CALL_ALARM = True

# 休眠时间
SLEEP_SHORT = 0.5
SLEEP_MEDIUM = 3
SLEEP_LONG = 6


# 最大重试次数和间隔
MAX_TRY = 3
RETRY_WAIT = 1
