import sys
import time
from random import random


def write_line(line):
    sys.stdout.write("\r" + line)
    sys.stdout.write("\n")
    sys.stdout.flush()


class Retry:
    def __init__(self, max_retries, retry_on):
        self.max_retries = max_retries
        self.retry_on = retry_on
        self.retries = 0

    def backoff(self):
        """
        Backoff for a random time based on number of retries.
        """
        base_t_ms = 100
        max_t_ms = 30000
        sleep_ms = random() * min(max_t_ms, base_t_ms * 2**self.retries)
        time.sleep(int(sleep_ms) / 1000)

    def __call__(self, func, *args, **kwargs):
        retry_on = self.retry_on
        if not retry_on:
            retry_on = (Exception,)
        self.retries = 0
        while self.retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                if isinstance(e, retry_on):
                    if self.retries >= self.max_retries:
                        raise
                    self.backoff()
                    self.retries += 1
                else:
                    raise
