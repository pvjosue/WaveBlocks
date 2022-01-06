# Python imports
import logging
from waveblocks.utils.logger import WaveblocksFormatter, MyLogger

# Configure Logging
waveblocks_logger = logging.getLogger("Waveblocks")
waveblocks_logger.setLevel(logging.INFO)
logger_handler = logging.StreamHandler()
logger_handler.setFormatter(WaveblocksFormatter())
waveblocks_logger.addHandler(logger_handler)

# Folders
from waveblocks.blocks import *
from waveblocks.utils import *
from waveblocks.microscopes import *
from waveblocks.reconstruction import *
from waveblocks.evaluation import *
