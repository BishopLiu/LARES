import importlib
from recbole.data.utils import create_dataset as create_recbole_dataset
import logging
import os
import colorlog
import re
import hashlib
from recbole.utils.utils import get_local_time, ensure_dir
from colorama import init
from recbole.utils.logger import RemoveColorFilter
log_colors_config = {
    "DEBUG": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


def create_dataset(config):
    dataset_module = importlib.import_module('data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        return create_recbole_dataset(config)

def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config-old)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = "./log/"
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    model_name = os.path.join(dir_name, config["model"])
    ensure_dir(model_name)
    config_str = "".join([str(key) for key in config.final_config_dict.values()])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    logfilename = "{}/{}-{}-{}-{}-{}.log".format(
        config["model"], config["model"], config["dataset"], config['train_stage'], get_local_time(), md5
    )

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config["state"] is None or config["state"].lower() == "info":
        level = logging.INFO
    elif config["state"].lower() == "debug":
        level = logging.DEBUG
    elif config["state"].lower() == "error":
        level = logging.ERROR
    elif config["state"].lower() == "warning":
        level = logging.WARNING
    elif config["state"].lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])