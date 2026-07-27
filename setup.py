import sys

import utils


def setup():
    with utils.announce("Running the setup"):
        print("Placeholder")


def log(msg):
    print(msg, file=sys.stderr)


