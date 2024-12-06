import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler

log_path = 'log'
if not os.path.exists(log_path):
    os.makedirs(log_path)

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(filename)s - %(levelname)s - %(message)s",
                              datefmt='%m/%d/%Y %H:%M:%S')
formatter.converter = time.localtime

stream_handler = logging.StreamHandler(sys.stdout)  # Standard output
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

rotating_handler = TimedRotatingFileHandler('log/app.log', when='midnight', backupCount=30)  # Daily rotating log file
rotating_handler.setFormatter(formatter)
rotating_handler.setLevel(logging.INFO)

logger = logging.getLogger()

logger.addHandler(stream_handler)
logger.addHandler(rotating_handler)
logger.setLevel(logging.INFO)
