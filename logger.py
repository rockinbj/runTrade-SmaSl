import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from os import path

from settings import LOG_LEVEL_CONSOLE, LOG_LEVEL_FILE, LOG_PATH

# 定义log文件的存储路径和名称
logPath = LOG_PATH
logName = "log.current"
logFile = path.join(logPath, logName)

# 定义logger对象
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

# 定义logger的记录格式
strFmt = "%(asctime)s|%(name)-10s:%(lineno)4d|%(processName)-12s|%(levelname)-8s %(message)s"
logFmt = logging.Formatter(strFmt)

# 定义屏幕输出
hConsole = logging.StreamHandler()
hConsole.setLevel(LOG_LEVEL_CONSOLE)
hConsole.setFormatter(logFmt)

# 定义文件输出
hFile = ConcurrentRotatingFileHandler(logFile, maxBytes=1024*1024*10, backupCount=30, encoding="utf-8")
hFile.setLevel(LOG_LEVEL_FILE)
hFile.setFormatter(logFmt)

# 将输出添加到logger
logger.addHandler(hConsole)
logger.addHandler(hFile)
