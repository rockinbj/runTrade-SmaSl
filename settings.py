# 策略命名
RUN_NAME = "鹤-offset-Test"
EXCHANGE_ID = "binance"

IS_TEST = True
SKIP_TRADE = True

# 轮动池黑白名单,格式"BTC/USDT"
SYMBOLS_WHITE = []
SYMBOLS_BLACK = ["BTCDOM/USDT", "DEFI/USDT", "BLUEBIRD/USDT"]

# 选币范围
# 只选取USDT交易对的成交量topN
RULE = "/USDT"
TYPE = "quoteVolume"
TOP = 60

# 选币个数
SELECTION_NUM = 1
# 是允许否重复加仓，即多个offset选中同一个币
# unfinished
OVERLAPS = False

# 开仓参数，4h*3，例如12小时bias因子涨幅排行
OPEN_FACTOR = "s_bias"
OPEN_LEVEL = "1h"
OPEN_PERIOD = 6

# 持仓时间，offset
HOLD_TIME = "48h"
OFFSET_LIST = [8,32]

# 平仓参数，例如4h级别close小于MA30
CLOSE_LEVEL = "4h"
CLOSE_FACTOR = "s_sma"
CLOSE_PERIODS = [20]  # 必须是list, method=="less"放一个元素, method="sma1LtSma2"放两个
CLOSE_METHOD = "less"
# CLOSE_METHOD = "sma1LtSma2"

# 过滤因子，涨幅下限，小于此则排除
FILTER_FACTORS = {
    "f_bias": {
        "level": OPEN_LEVEL,
        "period": OPEN_PERIOD,
        "base": 0 / 100,
        "gt": True,
    },
    # "f_ema_trend": {
    #     "level": CLOSE_LEVEL,
    #     "periods": [20,60,120],
    #     "long": True,
    # },
    "f_sma": {
        "level": CLOSE_LEVEL,
        "period": CLOSE_PERIODS[0],
        "gt": True,
    },
}

# 页面杠杆率
PAGE_LEVERAGE = 2
# 资金上限，控制实际杠杆率
MAX_BALANCE = 50 / 100
# 保证金模式
MARGIN_TYPE = "CROSSED"
# MARGIN_TYPE = "ISOLATED"
# 限价单安全滑点
SLIPPAGE = 1.5 / 100
# 最小下单金额，更小忽略
MIN_PAYMENT = 7

# 获取最新k线的数量
# 有使用EMA的计算需要大量k线让长均线(120)收敛到接近值
NEW_KLINE_NUM = 200
# 本轮开始之前的预留秒数，小于预留秒数则顺延至下轮
AHEAD_SEC = 3
# 向前偏移秒数，为了避免在同一时间下单造成踩踏
OFFSET_SEC = 0

# 报告发送间隔分钟
REPORT_INTERVAL = "30m"
# 重要告警是否拨打电话
CALL_ALARM = True

# 设置Log
LOG_LEVEL_CONSOLE = "debug"
LOG_LEVEL_FILE = "debug"
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent
LOG_PATH = ROOT_PATH / "log"

# 休眠时间
SLEEP_SHORT = 0.5
SLEEP_MEDIUM = 3
SLEEP_LONG = 6

# 最大重试次数和间隔
MAX_TRY = 3
RETRY_WAIT = 1
