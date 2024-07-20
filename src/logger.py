import logging
import os
from datetime import datetime

# define log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# setting path for logs
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

# creating logs directory
os.makedirs(logs_path,exist_ok=True)

# setting full log file path
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


if __name__=="__main__":
    logging.info("logging has been started")


