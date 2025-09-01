# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os

def setup_logger(log_file='table_provider.log') -> logging.Logger:
    # 创建日志目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 创建 Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 创建控制台处理程序
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # 创建文件处理程序
    file_handler = logging.FileHandler(os.path.join('logs', log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
