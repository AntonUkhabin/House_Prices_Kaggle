import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    '''Fix random seeds for reproducible results.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)



class TeeLogger:
    '''Write terminal output both to console and log file.'''

    def __init__(self, log_path):
        # Save original terminal output stream.
        self.terminal = sys.stdout
        self.original_stderr = sys.stderr

        # Open log file for writing.
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        '''Return whether the terminal stream is interactive.'''

        return self.terminal.isatty()

    def close(self):
        self.log_file.close()


def setup_run_logging(config):
    '''Redirect terminal output to console and run log file.'''

    logs_dir = Path(config.paths.path_to_logs)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f'{config.general.experiment_name}.txt'

    tee_logger = TeeLogger(log_path)

    # Replace default stdout/stderr so every print() is written both to terminal and log file.
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    return tee_logger