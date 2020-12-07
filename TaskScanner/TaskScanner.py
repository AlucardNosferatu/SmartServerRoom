import os
import json
import threading
import time
import datetime
import portalocker

task_list = {}


# 用于处理单个任务，统一抛出异常
# 如无异常返回OK
def task_process(file_name):
    try:
        with open(os.path.join('../Tasks', file_name), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json_dict = json.load(f)
        print(json_dict)
        ret = 'OK'
    except Exception as e:
        print(repr(e))
        ret = str(e)
    task_list[file_name]['ret'] = ret
    task_list[file_name]['status'] = 'stopped'


# 解析任务，根据任务状态决定任务文件的备份与销毁
# 任务失败后自动重试
def task_parse(file_name):
    # 建立任务状态字典
    if file_name not in task_list:
        print('first trial')
        task_list[file_name] = {
            'status': 'wait',
            'start_time': str(datetime.datetime.now()),
            'tried': 0,
            'ret': ''
        }
    else:
        print('tried file')

    # 有限次数重试失败的任务
    while task_list[file_name]['tried'] < 5:
        # 这里之后用线程池进行改造，抛出仅在子线程
        task_list[file_name]['status'] = 'ongoing'

        # 用新线程执行原子能力
        tp = threading.Thread(target=task_process, args=(file_name,))
        tp.start()
        # task_process(file_name)

        while task_list[file_name]['status'] != 'stopped':
            time.sleep(0.5)
            print('wait until process being finished')

        task_list[file_name]['tried'] += 1
        # ret代表任务单次的结果，之后也用管道文件来进行异步传递

        if task_list[file_name]['ret'] == 'OK':
            task_list[file_name]['status'] = 'completed'
            break
        else:
            task_list[file_name]['status'] = 'failed'
    try:
        if task_list[file_name]['status'] == 'completed':
            print('completed')
        else:
            print('failed')
            with open(os.path.join('../Tasks', file_name), 'r') as fr:
                portalocker.lock(fr, portalocker.LOCK_EX)
                lines = fr.readlines()
                with open(os.path.join('Failed', file_name), 'w') as fw:
                    fw.writelines(lines)
        os.remove(os.path.join('../Tasks', file_name))
    except Exception as e:
        print('error while dumping failed task file:', file_name)
        print(repr(e))
    del task_list[file_name]


def main_loop():
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join('console.txt'), 'w') as f:
        f.writelines([str(pid)])
    old_files = []
    flag = True
    while flag:
        time.sleep(1)
        print('Scan new task files...')
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
            else:
                cmd = ''
        flag = (cmd != 'stop')
    print('Scan loop ended normally.')


if __name__ == '__main__':
    main_loop()
