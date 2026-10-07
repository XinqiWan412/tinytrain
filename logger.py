import os


log_file = None


def setup_log(path):
    global log_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file = open(path, "a")


def log(text):
    print(text)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()
    