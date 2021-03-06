import os
import time
import datetime

AtD = 'FaceRec/DBFace/atomic_pid.txt'
ApF = 'FaceRec/FaceRec/Faces_Temp/app_pid.txt'
AtF = 'FaceRec/FaceRec/Faces_Temp/atomic_pid.txt'
ApM = 'MotionDet/M_Temp/app_pid.txt'
AtM = 'MotionDet/M_Temp/atomic_pid.txt'
ApP = 'PedestrianDet/P_Temp/app_pid.txt'
AtP = 'PedestrianDet/P_Temp/atomic_pid.txt'

pid_path = [AtD, ApF, AtF, ApM, AtM, ApP, AtP]
log_path = 'pid.txt'

if __name__ == '__main__':
    pid_list = ['0'] * len(pid_path)
    while True:
        for i in range(len(pid_path)):
            with open(pid_path[i], 'r') as f:
                pid_list[i] = f.readlines()[0].strip()
        process = os.popen('ps -ef | grep python')
        current_pid_list = process.readlines()
        alive = [False] * len(pid_path)
        for i, pid in enumerate(pid_list):
            for current_pid in current_pid_list:
                if pid in current_pid:
                    alive[i] = True
                    break
        info = [
            ' '.join(
                [
                    str(datetime.datetime.now()),
                    str(alive) + '\n'
                ]
            )
        ]
        info += [
            ' '.join(
                [
                    str(datetime.datetime.now()),
                    str(pid_list) + '\n'
                ]
            )
        ]
        with open(log_path, 'a') as f:
            f.writelines(info)
        time.sleep(120)
