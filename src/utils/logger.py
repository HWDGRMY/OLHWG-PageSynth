#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: logger.py
@description: 统一的日志配置。
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "olhwg",
                 log_file: str = None,
                 level: int = logging.INFO) -> logging.Logger:
    """配置并返回指定名称的 logger。

    Args:
        name: logger 名称。
        log_file: 若提供，将日志同时写入该文件。
        level: 日志级别。

    Returns:
        配置好的 logger 对象。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:  # 避免重复添加 handler
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger