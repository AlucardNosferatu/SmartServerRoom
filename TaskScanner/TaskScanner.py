import os
import json
import time
import datetime
import portalocker

task_list = {}


def task_process(file_name):
    try:
        with open(os.path.join('../Tasks', file_name), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json_dict = json.load(f)
        print(json_dict)
        return 'OK'
    except Exception as e:
        print(repr(e))
        return str(e)


def task_parse(file_name):
    if file_name not in task_list:
        print('first trial')
        task_list[file_name] = {
            'status': 'wait',
            'start_time': str(datetime.datetime.now()),
            'tried': 0
        }
    else:
        print('tried file')

    while task_list[file_name]['tried'] < 5:
        ret = task_process(file_name)
        task_list[file_name]['tried'] += 1
        if ret == 'OK':
            task_list[file_name]['status'] = 'completed'
            break
        else:
            task_list[file_name]['status'] = 'failed'

    if task_list[file_name]['status'] == 'completed':
        print('completed')
    else:
        print('failed')
        try:
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
            if len(cmd_list) > 0:
                cmd = cmd_list[0].strip()
            else:
                cmd = ''
        flag = (cmd != 'stop')
    print('Scan loop ended normally.')


if __name__ == '__main__':
    main_loop()
