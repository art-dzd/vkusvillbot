import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # httpx на INFO логирует полные URL, включая токен Telegram в /bot<TOKEN>/...
    # По умолчанию глушим, чтобы секреты не утекали в логи.
    if logging.getLogger("httpx").level in (logging.NOTSET, logging.INFO):
        logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_dialog_logger(log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger("dialog")
    if logger.handlers:
        return logger
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / "dialog.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
