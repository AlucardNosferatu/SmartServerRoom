import os
import time

AtD = 'FaceRec/DBFace/atomic_pid.txt'
ApF = 'FaceRec/FaceRec/Faces_Temp/app_pid.txt'
AtF = 'FaceRec/FaceRec/Faces_Temp/atomic_pid.txt'
ApM = 'MotionDet/M_Temp/app_pid.txt'
AtM = 'MotionDet/M_Temp/atomic_pid.txt'
ApP = 'PedestrianDet/P_Temp/app_pid.txt'
AtP = 'PedestrianDet/P_Temp/atomic_pid.txt'

if __name__ == '__main__':
    AtD_pid = 0
    ApF_pid = 0
    AtF_pid = 0
    ApM_pid = 0
    AtM_pid = 0
    ApP_pid = 0
    AtP_pid = 0
    while True:
        with open(AtD, 'r') as f:
            AtD_pid = int(f.readlines()[0].strip())
        print('AtD_pid', AtD_pid)
        time.sleep(5)
