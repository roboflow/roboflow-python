import os
import yaml

from typing import Callable


def amend_data_yaml(path: str, callback: Callable[[dict], dict]):
    with open(path) as source:
        content = yaml.safe_load(source)
    content = callback(content)
    os.remove(path)
    with open(path, "w") as target:
        yaml.dump(content, target)
