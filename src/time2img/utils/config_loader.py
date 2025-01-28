#!/usr/bin/env python
from configparser import ConfigParser

def get_config(path: str) -> ConfigParser:
    config = ConfigParser()
    config.read(path, encoding='UTF-8')
    return config