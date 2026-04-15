import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[
        logging.FileHandler("app1.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ArthmeticApp")


def add(a,b):
    try:
        c = a+b
        logger.debug(f"sub {a} + {b} = {c}")
        return c
    except ZeroDivisionError:
        logger.error("Division by zero")
        return 0

add(10,20)
