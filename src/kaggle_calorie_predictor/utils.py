from datetime import datetime
import functools
import json
import logging
import logging.config
from time import perf_counter
import torch


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        logging.info(f"{func.__name__!r} finished: {elapsed_time:0.4f}s")
        return value

    return wrapper_timer


def date_filename(prefix: str, file_type: str, include_time: bool = False) -> str:
    timestamp = datetime.now().strftime("%y%m%d_%H-%M-%S")
    datestamp = datetime.now().strftime("%y%m%d")
    filename = f"{prefix}_{timestamp if include_time else datestamp}{file_type}"
    return filename


def init_logging(model_name=None):
    logfile = f"logs/{date_filename(f'{model_name}-', '.log', include_time=True)}"
    logging.config.fileConfig(
        "./logging.ini",
        disable_existing_loggers=False,
        defaults={"logfilename": logfile},
    )
    return logfile


def get_parameters(path):
    with open(path, "r") as f:
        params = json.load(f)
    hp = params.get("hyperparameters", {})
    params.pop("hyperparameters", None)
    hp["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return params, hp
