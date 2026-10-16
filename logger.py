import os
from parallel.dist_utils import is_main


log_file = None


def setup_log(path):
    global log_file
    if not is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file = open(path, "a")


def log(text):
    if not is_main():
        return
    print(text)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()
    