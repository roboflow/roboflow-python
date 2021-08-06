import requests


def check_image_url(url):
    if 'http://' in url or "https://" in url:
        r = requests.head(url)
        return r.status_code == requests.codes.ok

    return False
