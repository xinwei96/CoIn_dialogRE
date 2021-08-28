import random
import numpy as np
import torch
import time
import logging


def init_logger(path):
    timestamp = time.time()
    log_path = path
    logger = logging.getLogger("mainModule")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def calculate_metric(correct_nums, pred_nums, target_nums):
    precision = 1 if pred_nums == 0 else correct_nums / pred_nums
    recall = 0 if target_nums == 0 else correct_nums / target_nums
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f_1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def print_message(message, logger, local_rank=0):
    if local_rank == 0:
        print(message)
        logger.info(message)