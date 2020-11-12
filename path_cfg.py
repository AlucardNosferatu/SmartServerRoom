import os
import sys

fr_path = os.path.abspath('FaceRec/FaceRec')
if fr_path not in sys.path:
    print('Add FR path')
    sys.path.append(fr_path)

md_path = os.path.abspath('MotionDet')
if md_path not in sys.path:
    print('Add MD path')
    sys.path.append(md_path)

pd_path = os.path.abspath('PedestrianDet')
if pd_path not in sys.path:
    print('Add PD path')
    sys.path.append(pd_path)

print('Path imported')
