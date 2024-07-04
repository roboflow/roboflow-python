import sys


def write_line(line):
    sys.stdout.write("\r" + line)
    sys.stdout.write("\n")
    sys.stdout.flush()


class Retry:
    def __init__(self, max_retries, retry_on):
        self.max_retries = max_retries
        self.retry_on = retry_on
        self.retries = 0

    def __call__(self, func, *args, **kwargs):
        self.retries = 0
        retry_on = self.retry_on
        if not retry_on:
            retry_on = (Exception,)
        self.retries = 0
        while self.retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                if isinstance(e, retry_on):
                    self.retries += 1
                    if self.retries > self.max_retries:
                        raise
                else:
                    raise
