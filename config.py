import os
import torch
from recbole.config import Config as RecBoleConfig


class Config(RecBoleConfig):
    def __init__(self, config):
        super().__init__(config)
