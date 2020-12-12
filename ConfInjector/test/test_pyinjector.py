import time

from pyinjector import inject

INJECTION_LIB_PATH = 'injection/build/lib.win-amd64-3.7/injection.cp37-win_amd64.pyd'
TIME_TO_WIT_FOR_PROCESS_TO_INIT = 1
TIME_TO_WIT_FOR_INJECTION_TO_RUN = 1


def test_inject():
    try:
        time.sleep(TIME_TO_WIT_FOR_PROCESS_TO_INIT)
        inject(int(input()), INJECTION_LIB_PATH)
        time.sleep(TIME_TO_WIT_FOR_INJECTION_TO_RUN)
    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    test_inject()