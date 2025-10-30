from rich.logging import RichHandler
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler()]
    )

    logger = logging.getLogger('uvicorn')
    logger.setLevel(logging.INFO)
    return logging.getLogger(__name__)
