import logging
from rich.logging import RichHandler


def get_logger(name, level=logging.DEBUG):
    for mod in ['numba', 'matplotlib']:
        logging.getLogger(mod).setLevel(logging.CRITICAL)

    logging.basicConfig(
        level=level,
        format='Line %(lineno)d: %(name)s: %(message)s',
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )
    logger = logging.getLogger(name)
    return logger
