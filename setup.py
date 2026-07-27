import sys
import requests

import utils
URL_ANIMALS = r"https://gist.github.com/raineorshine/599777e98e5e968a15c545043973f035/raw"


def setup():
    utils.log("Debug setup")
    result_of_request = requests.get(URL_ANIMALS)
    print(result_of_request.content)








