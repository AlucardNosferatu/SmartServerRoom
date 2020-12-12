import sys
import time
from subprocess import Popen, PIPE
from importlib.util import find_spec
from pyinjector import inject

INJECTION_LIB_PATH = find_spec('injection').origin
STRING_PRINTED_FROM_LIB = b'Hello, world!'
TIME_TO_WIT_FOR_PROCESS_TO_INIT = 1
TIME_TO_WIT_FOR_INJECTION_TO_RUN = 1


def test_inject():
    try:
        time.sleep(TIME_TO_WIT_FOR_PROCESS_TO_INIT)
        pid_override = input()
        inject(int(pid_override), INJECTION_LIB_PATH)
        time.sleep(TIME_TO_WIT_FOR_INJECTION_TO_RUN)
    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    test_inject()
