import os
import json
import threading
import time
import datetime
import portalocker
from logger_TS import logger

task_list = {}


# 用于处理单个任务，统一抛出异常
# 如无异常返回OK
def task_process(file_name):
    try:
        with open(os.path.join('../Tasks', file_name), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json_dict = json.load(f)
        print(json_dict)
        logger.debug(str(json_dict))
        ret = 'OK'
        # Only for test
        # assert ret != 'OK'
    except Exception as e:
        print(repr(e))
        logger.error(repr(e))
        ret = repr(e)
    task_list[file_name]['ret'] = ret
    task_list[file_name]['status'] = 'stopped'


# 解析任务，根据任务状态决定任务文件的备份与销毁
# 任务失败后自动重试
def task_parse(file_name):
    # 建立任务状态字典
    if file_name not in task_list:
        print('first trial')
        logger.info('first trial')
        task_list[file_name] = {
            'status': 'wait',
            'start_time': str(datetime.datetime.now()),
            'tried': 0,
            'ret': ''
        }
    else:
        print('tried file')
        logger.info('tried file')

    # 有限次数重试失败的任务
    while task_list[file_name]['tried'] < 5:
        # 这里之后用线程池进行改造，抛出仅在子线程
        task_list[file_name]['status'] = 'ongoing'

        # 用新线程执行原子能力
        tp = threading.Thread(target=task_process, args=(file_name,))
        tp.start()
        # task_process(file_name)

        count = 0
        while task_list[file_name]['status'] != 'stopped':
            time.sleep(0.2)
            count += 1
            if count % 10 == 0:
                print('wait until process being finished')
                logger.info('wait until process being finished')
                count = 0

        task_list[file_name]['tried'] += 1
        # ret代表任务单次的结果，之后也用全局变量来进行异步传递
        # 根据任务结果给任务状态赋值
        if task_list[file_name]['ret'] == 'OK':
            task_list[file_name]['status'] = 'completed'
            break
            # 如果第一次就成功则退出重试循环
        else:
            task_list[file_name]['status'] = 'failed'

    try:
        if task_list[file_name]['status'] == 'completed':
            print('completed')
            logger.info('completed')
        else:
            print('failed')
            logger.info('failed')
            # 若失败，则将请求与处理结果打包放进记录文件
            with open(os.path.join('../Tasks', file_name), 'r') as fr:
                portalocker.lock(fr, portalocker.LOCK_EX)
                task_jd = json.load(fr)
                with open(os.path.join('Failed', file_name), 'w') as fw:
                    json.dump({'result': task_list[file_name], 'request': task_jd}, fw)
        # 无论是否成功都将请求文件删除
        os.remove(os.path.join('../Tasks', file_name))
    except Exception as e:
        print('error while dumping failed task file:', file_name)
        print(repr(e))
        logger.error('error while dumping failed task file:' + ' ' + file_name)
        logger.error(repr(e))
    del task_list[file_name]


def main_loop():
    pid = os.getpid()
    print('pid is:', pid)
    logger.debug('pid is:'+' '+str(pid))
    with open(os.path.join('console.txt'), 'w') as f:
        f.writelines([str(pid)])
    old_files = []
    flag = True
    while flag:
        time.sleep(1)
        print('Scan new task files...')
        # logger.info('Scan new task files...')
        current_files = os.listdir('../Tasks')
        new_files = list(set(current_files).difference(set(old_files)))
        for file in new_files:
            task_parse(file)
        old_files = current_files
        with open('console.txt', 'r') as f:
            cmd_list = f.readlines()
            if len(cmd_list) > 1:
                cmd = cmd_list[1].strip()
                print('cmd detected:', cmd)
                logger.debug('cmd detected:' + ' ' + cmd)
            else:
                cmd = ''
        flag = (cmd != 'stop')
    print('Scan loop ended normally.')
    logger.info('Scan loop ended normally.')


if __name__ == '__main__':
    main_loop()
