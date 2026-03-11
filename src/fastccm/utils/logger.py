import logging
import sys
from datetime import datetime


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def setup_logger(name: str, verbose: int = 0, log_file: str = None) -> logging.Logger:
    """
    verbose:
      0 -> WARNING
      1 -> INFO
      2+-> DEBUG

    If log_file is None -> log to terminal (stderr).
    Else -> log to file.
    """
    logger = logging.getLogger(name)

    level = logging.WARNING if verbose <= 0 else (logging.INFO if verbose == 1 else logging.DEBUG)
    logger.setLevel(level)

    # Avoid duplicate handlers across multiple PairwiseCCM instances.
    if not getattr(logger, "_pairwiseccm_configured", False):
        logger.propagate = False

        fmt = MicrosecondFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f",
        )

        handler = logging.StreamHandler(sys.stderr) if log_file is None else logging.FileHandler(log_file)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        logger._pairwiseccm_configured = True
    else:
        # Update handler levels when verbose changes.
        for h in logger.handlers:
            h.setLevel(level)

    return logger
