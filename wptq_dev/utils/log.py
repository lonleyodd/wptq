import os
import torch
import logging
import coloredlogs
from functools import partial

def get_logger():
    root='./log'
    os.makedirs(root,exist_ok=True)

    log_path=os.path.join(root,"quant.log")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  

    ch = logging.StreamHandler()  
    fh = logging.FileHandler(log_path,'w')

    ch.setLevel(logging.INFO)  
    fh.setLevel(logging.DEBUG)

    logger.addHandler(ch)  
    logger.addHandler(fh)

    coloredlogs.install(level='INFO',
                        logger=logger,
                        fmt='%(asctime)s %(name)s %(message)s')
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)

    logger.info('output and logs will be saved to {}'.format(log_path))
    return logger




