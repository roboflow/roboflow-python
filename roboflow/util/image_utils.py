import requests


# Checks if hosted image hurl is a valid image
def check_image_url(url):
    if "http://" in url or "https://" in url:
        r = requests.head(url)
        return r.status_code == requests.codes.ok

    return False
