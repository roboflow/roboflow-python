"""HTTP requests made by the SDK.

These wrap the ``requests`` functions of the same name and add the SDK's User-Agent, so the
Roboflow API can tell SDK traffic apart from other Python clients.
"""

from typing import Any, Dict

import requests

from roboflow import __version__

USER_AGENT = f"roboflow-python/{__version__}"


def _with_user_agent(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Headers passed by the caller take precedence.
    kwargs["headers"] = {"User-Agent": USER_AGENT, **(kwargs.get("headers") or {})}
    return kwargs


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    return requests.request(method, url, **_with_user_agent(kwargs))


def get(url: str, **kwargs: Any) -> requests.Response:
    return requests.get(url, **_with_user_agent(kwargs))


def post(url: str, **kwargs: Any) -> requests.Response:
    return requests.post(url, **_with_user_agent(kwargs))


def put(url: str, **kwargs: Any) -> requests.Response:
    return requests.put(url, **_with_user_agent(kwargs))


def patch(url: str, **kwargs: Any) -> requests.Response:
    return requests.patch(url, **_with_user_agent(kwargs))


def delete(url: str, **kwargs: Any) -> requests.Response:
    return requests.delete(url, **_with_user_agent(kwargs))


def head(url: str, **kwargs: Any) -> requests.Response:
    return requests.head(url, **_with_user_agent(kwargs))


class Session(requests.Session):
    """A ``requests.Session`` that sends the SDK's User-Agent."""

    def __init__(self) -> None:
        super().__init__()
        self.headers["User-Agent"] = USER_AGENT
