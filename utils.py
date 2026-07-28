import time
from contextlib import ContextDecorator
import sys
from urllib.request import urlopen

def log(msg):
    print(msg, file=sys.stderr)


class ExecutionTime(ContextDecorator):
    start_time: float
    end_time: float
    runtime: float

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time

    def __str__(self):
        return f"{self.runtime:.6f}"


def execution_timer():
    return ExecutionTime()


class Announce(ContextDecorator):
    action_str: str
    timer: ExecutionTime
    verbose: bool

    def __init__(self, action_str: str, verbose=True):
        self.action_str = action_str
        self.timer = ExecutionTime()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(f"starting [{self.action_str}]")
        self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        runtime = self.timer.runtime
        if self.verbose:
            print(f"[{self.action_str}]...Finished (took {runtime:2f} seconds)")


def announce(action: str, verbose=True):
    return Announce(action, verbose)

def download_raw_text_from_website(url: str) -> str:
    # Without any error prevention...
    with urlopen(url) as resp:
        result_text = resp.read().decode("utf-8", errors="replace")
    return result_text

def second(x):
    return x[1]