import os
import sys
import time
import zipfile
from random import random

from requests import Response
from tqdm import tqdm

from roboflow.config import TQDM_DISABLE


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


def save_response_content(response: Response, path: str, desc: str | None = None) -> None:
    """Write an HTTP response in bounded chunks with byte-based progress.

    Args:
        response: Streaming response whose HTTP status has already been checked.
        path: Destination file path.
        desc: Description shown in the progress bar.
    """
    total: int | None
    try:
        total = int(response.headers.get("content-length", ""))
        if total < 0:
            total = None
    except (TypeError, ValueError):
        total = None
    # iter_content decodes compressed responses; Content-Length counts wire bytes.
    if (
        response.headers.get("content-encoding", "identity").lower() != "identity"
        or "transfer-encoding" in response.headers
    ):
        total = None
    # urllib3 1.x does not enforce Content-Length by default, including for gzip.
    response.raw.enforce_content_length = True
    with (
        open(path, "wb") as f,
        tqdm(total=total, desc=desc, unit="B", unit_scale=True, disable=bool(TQDM_DISABLE)) as progress,
    ):
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))


def extract_zip(location: str, desc: str = "Extracting"):
    """Extract ``roboflow.zip`` inside *location* and remove the archive.

    Args:
        location: Directory containing ``roboflow.zip``.
        desc: Description shown in the tqdm progress bar.
    """
    zip_path = os.path.join(location, "roboflow.zip")
    tqdm_desc = None if TQDM_DISABLE else desc
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=tqdm_desc):
            try:
                zip_ref.extract(member, location)
            except zipfile.error:
                raise RuntimeError("Error unzipping download")

    os.remove(zip_path)
