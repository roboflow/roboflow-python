import sys


def write_line(line):
    sys.stdout.write("\r" + line)
    sys.stdout.write("\n")
    sys.stdout.flush()


def retry(max_retries, retry_on, func, *args, **kwargs):
    if not retry_on:
        retry_on = (Exception,)
    retries = 0
    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            if isinstance(e, retry_on):
                retries += 1
                if retries > max_retries:
                    raise
            else:
                raise
