import logging
from logging import getLoggerClass, setLoggerClass, NOTSET
from colorama import Fore


class WaveblocksFormatter(logging.Formatter):
    format_simple = "Waveblocks - %(levelname)s - %(message)s"
    format_advanced = (
        "Waveblocks - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: Fore.MAGENTA + format_advanced + Fore.RESET,
        logging.INFO: format_simple,
        logging.WARNING: Fore.YELLOW + format_advanced + Fore.RESET,
        logging.ERROR: Fore.RED + format_advanced + Fore.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class MyLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)
        self.debug_mla = False
        self.debug_microscope = False
        self.debug_richardson_lucy = False
        self.debug_optimizer = False
        self.bkplevel = logging.INFO

    def debug_override(self, text):
        self.bkplevel = self.level
        self.setLevel(logging.DEBUG)
        self.debug(text, stacklevel=2)
        self.setLevel(self.bkplevel)


def set_logging(debug_mla=False, debug_microscope=False, debug_richardson_lucy=False, debug_optimizer=False):
    logger = logging.getLogger("Waveblocks")
    logger.debug_mla = debug_mla
    logger.debug_microscope = debug_microscope
    logger.debug_richardson_lucy = debug_richardson_lucy
    logger.debug_optimizer = debug_optimizer


setLoggerClass(MyLogger)
