import logging as _learnscapes_logging
import sys as _learnscapes_sys

logger = _learnscapes_logging.getLogger("learnscapes")
global_handler = _learnscapes_logging.StreamHandler(_learnscapes_sys.stdout)
logger.addHandler(global_handler)
logger.setLevel(_learnscapes_logging.DEBUG)