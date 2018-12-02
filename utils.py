import logging

logger = None
def get_logger(name=None):
    """return a logger
    """
    global logger
    if logger is not None: return logger
    print('Creating logger========================================>')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]{%(pathname)s:%(lineno)d} %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
