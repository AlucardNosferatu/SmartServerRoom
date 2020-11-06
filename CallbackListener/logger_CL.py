import logging
import os

from cfg_CL import save_path

log_path = os.path.join(save_path, 'CL.txt')
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_path,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(levelname)s - %(thread)d - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
)
logger = logging.getLogger('CL')