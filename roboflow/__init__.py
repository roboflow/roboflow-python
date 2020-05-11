import time
import requests

API_URL = "https://api.roboflow.ai"
_token = None
_token_expires = 0

def auth(api_key):
    global _token

    response = requests.post(API_URL + "/token", data=({
        "api_key": api_key
    }))

    r = response.json();
    if "error" in r:
        raise RuntimeError(response.text)

    _token = r["token"]
    _token_expires = time.time() + r["expires_in"]

    return r

def dataset(name):
    global _token

    if not _token:
        raise Exception("You must first auth with your API key to call this method.")

    response = requests.get(API_URL + "/dataset/" + name, params=({
        "access_token": _token
    }))

    r = response.json();
    if "error" in r:
        raise RuntimeError(response.text)

    return r

def load(dataset, *args):
    print(f"loading {dataset} {args}")
