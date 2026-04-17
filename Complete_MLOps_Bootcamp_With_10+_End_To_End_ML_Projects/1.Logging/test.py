from logging_script import logger_get

logging = logger_get()

def add(a,b):

    logging.info("The addition operation is taking place")

    c = a+b
    
    return c

add(10,15)





